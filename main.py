# ==============================================================================
# Final main.py for HackRx 6.0 - Intelligent Query-Retrieval System
# ==============================================================================

import os
import uuid
import requests
import asyncio
import time
from io import BytesIO
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from pypdf import PdfReader
import json
import re

# --- Core Libraries ---
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv

# --- AI & Vector DB Services ---
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

# --- PDF Processing & OCR ---
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image


# --- 1. Configuration & Clients ---

# Load environment variables from .env file
load_dotenv()

# API Key Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Configure Google Generative AI
genai.configure(api_key=GEMINI_API_KEY)

# Define embedding model and its dimension
EMBEDDING_MODEL = "models/embedding-001"
EMBEDDING_DIM = 768

# Initialize Pinecone and in-memory cache
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = None 
document_cache = {}


# --- 2. FastAPI App Initialization & Pydantic Models ---

app = FastAPI(title="HackRx 6.0 Query-Retrieval System")

class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]


# --- 3. Security ---

security = HTTPBearer()
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
    return True


# --- 4. Document Processing ---

# --- 4. Document Processing ---

def extract_text_from_pdf_stream(file_stream: BytesIO) -> str:
    """
    Intelligently extracts text from a PDF. It first tries a fast direct extraction.
    If that fails (indicating a scanned/image-based PDF), it falls back to OCR.
    """
    text = ""
    # Create two in-memory copies of the file stream to avoid consuming it
    stream_copy1 = BytesIO(file_stream.getvalue())
    stream_copy2 = BytesIO(file_stream.getvalue())

    # --- Method 1: Fast Direct Text Extraction ---
    try:
        print("Attempting fast direct text extraction...")
        pdf_reader = PdfReader(stream_copy1)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    except Exception:
        text = "" # Reset text if pypdf fails for any reason

    # If direct extraction yields significant text, use it.
    if len(text.strip()) > 100:
        print("Direct text extraction successful.")
        return text

    # --- Method 2: Fallback to OCR ---
    print("Direct extraction failed or yielded minimal text. Falling back to OCR...")
    return extract_text_with_ocr(stream_copy2)


def extract_text_with_ocr(file_stream: BytesIO) -> str:
    """
    The batched parallel OCR function (now used as a fallback).
    """
    all_texts = []
    try:
        images = convert_from_bytes(file_stream.read())
        print(f"PDF converted to {len(images)} page(s) for OCR processing.")

        batch_size = 4 # Process 4 pages at a time in parallel

        def ocr_page(image):
            return pytesseract.image_to_string(image) or ""

        with ThreadPoolExecutor() as executor:
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                print(f"Processing batch starting at page {i+1}...")
                texts = list(executor.map(ocr_page, batch))
                all_texts.extend(texts)
        
        print("Batched parallel OCR text extraction complete.")
        return "".join(all_texts)
        
    except Exception as e:
        print(f"Error extracting text via OCR: {e}")
        raise ValueError(f"Failed to extract text from PDF using OCR: {e}")
    
    
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Splits text into chunks with overlap."""
    if not text: return []
    words = text.split()
    if len(words) <= chunk_size: return [" ".join(words)]
    
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


# --- 5. Embedding, Vector Storage, and Retrieval ---

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates embeddings for a list of document texts using Gemini."""
    if not texts: return []
    response = genai.embed_content(model=EMBEDDING_MODEL, content=texts, task_type="RETRIEVAL_DOCUMENT")
    return response['embedding']

def upsert_chunks_to_pinecone(index, chunks: List[str], document_id: str):
    """Generates embeddings and upserts chunks to Pinecone."""
    if not chunks: return
    embeddings = generate_embeddings(chunks)
    vectors_to_upsert = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors_to_upsert.append({
            "id": f"{document_id}_chunk_{i}",
            "values": embedding,
            "metadata": {"text": chunk, "document_id": document_id}
        })
    
    for i in range(0, len(vectors_to_upsert), 100):
        index.upsert(vectors=vectors_to_upsert[i:i + 100])
    print(f"Successfully upserted {len(vectors_to_upsert)} vectors.")

def query_pinecone(index, query_embedding: List[float], top_k: int) -> List[str]:
    """Performs a semantic search in Pinecone."""
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [match.metadata['text'] for match in results.matches]

def hybrid_query(index, all_chunks: List[str], question: str, question_embedding: List[float], top_k: int) -> List[str]:
    """
    Combines semantic and keyword search, with limits to control context size.
    """
    # 1. Perform semantic search, reducing top_k for fewer results
    semantic_results = query_pinecone(index, question_embedding, top_k=4) # Reduced to 3
    
    # 2. Perform keyword search
    question_keywords = set(question.lower().split())
    keyword_results = [chunk for chunk in all_chunks if any(keyword in chunk.lower() for keyword in question_keywords)]
    
    # 3. Combine results, limiting keyword results to prevent excessive token usage
    # We take the best 3 semantic results and add up to 5 relevant keyword results
    combined_results = semantic_results + keyword_results[:6] # Limit keyword results
    
    unique_results = list(dict.fromkeys(combined_results))
    print(f"Found {len(unique_results)} unique chunks with hybrid search.")
    return unique_results


# --- 6. LLM Answer Generation ---

# async def get_answer_for_question(question: str, question_embedding: List[float], all_chunks: List[str], semaphore: asyncio.Semaphore):
#     """Asynchronously gets a single answer for a single question using a hybrid search context."""
#     async with semaphore:
#         print(f"Processing question: '{question[:]}...'")
        
#         context_chunks = hybrid_query(pinecone_index, all_chunks, question, question_embedding, top_k=5)
#         context_str = "\n---\n".join(context_chunks)
        
#         llm_model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
#         prompt = f"""You are a precise and meticulous insurance policy analyst. Your task is to answer a question based exclusively on the provided context.

#     **Instructions:**
#     1.  **Analyze the user's question** to understand what specific information is being asked.
#     2.  **Scan the provided context** to find all relevant clauses, definitions, limits, and waiting periods.
#     3.  **Think step-by-step**:
#         * First, explicitly state the rule or clause from the policy that applies to the question.
#         * Next, show the calculation if one is needed.
#         * Finally, provide a clear and concise final answer based on your step-by-step reasoning.
#     4.  If the relevant information is not available in the context, you MUST respond with the exact phrase: "The answer is not available in the provided document."
#     5.  Do not invent, assume, or infer any information not explicitly stated in the context.
#     6.  If the question is of Yes/No type, respond with "Yes" or "No" with a brief explanation based on the context.

#         **Context:**
#         ---
#         {context_str}
#         ---

#         **Question:** {question}

#         **JSON Response:**
#         ```json
#         """
        
#         response = await llm_model.generate_content_async(
#             prompt,
#             generation_config=genai.types.GenerationConfig(temperature=0.0)
#         )
        
#         final_answer = "Error: Could not parse LLM response." # Default error message
#         try:
#             # Clean up and parse the JSON response from the model
#             json_text = response.text.replace("```json", "").replace("```", "").strip()
#             parsed_response = json.loads(json_text)
#             final_answer = parsed_response.get("answer", "Error: 'answer' key not found in LLM response.")
#         except (json.JSONDecodeError, AttributeError):
#             # If the model fails to return valid JSON, use its raw text as a fallback
#             final_answer = response.text.strip()
        
#         # Print the final answer to the server logs
#         print(f"Final Answer Generated: {final_answer[:15]}")
        
#         return final_answer

async def get_answer_for_question(question: str, question_embedding: List[float], all_chunks: List[str], semaphore: asyncio.Semaphore):
    """
    Asynchronously gets a single, highly accurate answer with a clean output format.
    The LLM performs step-by-step reasoning internally but only the final answer is returned.
    """
    async with semaphore:
        print(f"Processing question: '{question[:30]}...'")
        
        context_chunks = hybrid_query(pinecone_index, all_chunks, question, question_embedding, top_k=5)
        context_str = "\n---\n".join(context_chunks)
        
        llm_model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        # This new prompt asks the model to separate its reasoning from the final answer
        prompt = f"""
        You are a highly intelligent and detail-focused **Intelligent Query Retrieval RAG and Policy Analyst AI**, trained to interpret **any kind of insurance policy document** (health, life, motor, travel, etc.) with extreme precision.

        Your task is to answer the user's question using **only the content from the provided policy document**. Do not use external knowledge, assumptions, or prior experience.

        ---

        ### 🔍 What You Must Do:

        #### 1. Rephrase the Question Internally (Silently)
        - Internally generate **2-3 alternate versions** of the question to explore edge cases, semantic variations, or scenario reinterpretations.
        - Identify whether the question is asking for:
        - A direct fact
        - A yes/no eligibility
        - A scenario-based outcome
        - A computed benefit or refund

        #### 2. Deep Contextual Matching
        - Search the entire policy text for:
        - **Exact matches**
        - **Semantically similar clauses**, even if the headings or phrases are different
        - Clauses buried in exclusions, footnotes, or definitions

        #### 3. Always Check Eligibility Conditions
        Whenever a benefit, coverage, or refund is mentioned, check for:
        - Age limits or dependent rules
        - Waiting periods or claim history requirements
        - Policy type, plan tier, or product variant
        - Residency, geography, relationship status, or employment conditions
        - Maximum usage limits (e.g., “only once during policy term”)

        If the conditions are not met, clearly explain **why** the benefit is not applicable.

        #### 4. Perform Numerical Reasoning if Needed
        If the question involves **any** calculations:
        - Extract relevant limits, durations, thresholds, or rates from the document
        - Show step-by-step calculation (e.g., subtracting deductibles, applying percentage limits, computing refund slabs)
        - Be accurate and never guess. Use only values explicitly stated in the document.

        #### 5. Link Information from Multiple Sections
        If relevant details are split across sections (e.g., one section defines eligibility, another defines limits), combine them **logically** to form a unified answer.

        ---

        ### 📌 Golden Rules

        - Use **only** the uploaded document (`{context_str}`). If the information is not found in the document, respond:
        `"The answer is not available in the provided document."`
        - If the question **requires a yes or no**, always start your answer with `"Yes,"` or `"No,"`, followed by the explanation.
        - Every answer must be:
        - Self-contained
        - Fact-based
        - Written in **professional and formal tone**
        - Explain any reasoning used to reach the answer

        ---

        ### 🧾 Context (Policy Document Extracted Text):
        {context_str}

        ---

        ---

        ### 💬 JSON Response Format:
        ```json
        {{
        "final_answer": "<Answer goes here — concised, shortend, briefly reasoned, and policy-specific with atmost two lines. Start with Yes/No if relevant.>"
        }}
        ```
        """

        
        response = await llm_model.generate_content_async(
        prompt,
        generation_config=genai.types.GenerationConfig(temperature=0.0)
    )

    # Default fallback answer
    final_answer = "Error: Could not parse LLM response."

    try:
        # Extract JSON content from model's markdown-style response
        # Step 1: Use regex to extract JSON inside triple backticks
        match = re.search(r"```json\s*(\{.*?\})\s*```", response.text, re.DOTALL)
        if match:
            json_text = match.group(1).strip()
        else:
            # Fallback: Try extracting raw JSON (if not inside backticks)
            json_text = response.text.strip()

        # Step 2: Parse the JSON
        parsed_response = json.loads(json_text)

        # Step 3: Extract the final_answer
        final_answer = parsed_response.get("final_answer", "Error: 'final_answer' key not found.")
    except (json.JSONDecodeError, AttributeError) as e:
        # As last fallback, use raw text
        final_answer = f"Raw model output:\n{response.text.strip()}\n\nParsing error: {str(e)}"

    # Logging or returning final answer
    print(f"Final Answer Generated: {final_answer}")
    return final_answer

# --- 7. FastAPI Startup & Main Endpoint ---

@app.on_event("startup")
async def startup_event():
    """Initializes the Pinecone index on application startup."""
    global pinecone_index
    print("FastAPI application starting up...")
    if PINECONE_INDEX_NAME not in pinecone_client.list_indexes().names():
        pinecone_client.create_index(
            name=PINECONE_INDEX_NAME, dimension=EMBEDDING_DIM, metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME)
    print(f"Pinecone index '{PINECONE_INDEX_NAME}' is ready.")

@app.post("/api/v1/hackrx/run", response_model=AnswerResponse, dependencies=[Depends(verify_token)])
async def run_submission(request: RunRequest):
    """Main endpoint to process documents and answer questions."""
    start_time = time.time()
    
    print(f"Received request for document: {request.documents}")
    session_document_id = str(uuid.uuid4())
    
    if request.documents in document_cache:
        print("Found document text in cache. Skipping download and OCR.")
        document_text = document_cache[request.documents]
    else:
        print("Document not in cache. Processing now.")
        try:
            response = requests.get(request.documents)
            response.raise_for_status()
            document_text = extract_text_from_pdf_stream(BytesIO(response.content))
            if not document_text.strip():
                raise HTTPException(status_code=400, detail="Could not extract text from document.")
            document_cache[request.documents] = document_text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

    document_chunks = chunk_text(document_text)
    upsert_chunks_to_pinecone(pinecone_index, document_chunks, session_document_id)

    print("Embedding all questions in a single batch...")
    question_embeddings_response = genai.embed_content(
        model=EMBEDDING_MODEL, content=request.questions, task_type="RETRIEVAL_QUERY"
    )
    question_embeddings = question_embeddings_response['embedding']
    print("Batch embedding complete.")

    semaphore = asyncio.Semaphore(5)
    tasks = []
    for i, question in enumerate(request.questions):
        tasks.append(get_answer_for_question(question, question_embeddings[i], document_chunks, semaphore))
    
    results = await asyncio.gather(*tasks)

    pinecone_index.delete(filter={"document_id": session_document_id})
    print(f"Cleaned up vectors for session {session_document_id}.")
    
    duration = time.time() - start_time
    print(f"--- Request completed in {duration:.2f} seconds ---")
    return AnswerResponse(answers=results)

# To run this application locally, use the following command in your terminal:
# uvicorn main:app --reload  




# ==============================================================================
# Final main.py for HackRx 6.0 - Intelligent Query-Retrieval System
# ==============================================================================

# import os
# import uuid
# import requests
# import asyncio
# import time
# import json
# from io import BytesIO
# from typing import List, Dict
# from concurrent.futures import ThreadPoolExecutor

# # --- Core Libraries ---
# from fastapi import FastAPI, HTTPException, Depends, status
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from pydantic import BaseModel
# from dotenv import load_dotenv

# # --- AI & Vector DB Services ---
# import google.generativeai as genai
# from pinecone import Pinecone, ServerlessSpec

# # --- PDF Processing & OCR ---
# import pytesseract
# from pdf2image import convert_from_bytes
# from pypdf import PdfReader


# # --- 1. Configuration & Clients ---
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
# BEARER_TOKEN = os.getenv("BEARER_TOKEN")
# PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# genai.configure(api_key=GEMINI_API_KEY)

# EMBEDDING_MODEL = "models/embedding-001"
# EMBEDDING_DIM = 768

# pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
# pinecone_index = None 
# document_cache = {}


# # --- 2. FastAPI App & Models ---
# app = FastAPI(title="HackRx 6.0 Query-Retrieval System")
# class RunRequest(BaseModel):
#     documents: str
#     questions: List[str]
# class AnswerResponse(BaseModel):
#     answers: List[str]


# # --- 3. Security ---
# security = HTTPBearer()
# def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
#     if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
#         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
#     return True


# # --- 4. Document Processing ---
# def extract_text_from_pdf_stream(file_stream: BytesIO) -> str:
#     text = ""
#     stream_copy1 = BytesIO(file_stream.getvalue())
#     stream_copy2 = BytesIO(file_stream.getvalue())
#     try:
#         print("Attempting fast direct text extraction...")
#         pdf_reader = PdfReader(stream_copy1)
#         for page in pdf_reader.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text
#     except Exception:
#         text = ""
#     if len(text.strip()) > 100:
#         print("Direct text extraction successful.")
#         return text
#     print("Direct extraction failed. Falling back to OCR...")
#     return extract_text_with_ocr(stream_copy2)

# def extract_text_with_ocr(file_stream: BytesIO) -> str:
#     all_texts = []
#     try:
#         images = convert_from_bytes(file_stream.read())
#         batch_size = 4
#         with ThreadPoolExecutor() as executor:
#             for i in range(0, len(images), batch_size):
#                 batch = images[i:i + batch_size]
#                 texts = list(executor.map(lambda img: pytesseract.image_to_string(img) or "", batch))
#                 all_texts.extend(texts)
#         return "".join(all_texts)
#     except Exception as e:
#         raise ValueError(f"Failed to extract text from PDF using OCR: {e}")

# def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
#     if not text: return []
#     words = text.split()
#     if len(words) <= chunk_size: return [" ".join(words)]
#     chunks = []
#     for i in range(0, len(words), chunk_size - overlap):
#         chunks.append(" ".join(words[i:i + chunk_size]))
#     return chunks


# # --- 5. Retrieval & Vector DB Functions ---
# def generate_embeddings(texts: List[str], task_type: str) -> List[List[float]]:
#     if not texts: return []
#     response = genai.embed_content(model=EMBEDDING_MODEL, content=texts, task_type=task_type)
#     return response['embedding']

# def upsert_chunks_to_pinecone(index, chunks: List[str], document_id: str):
#     if not chunks: return
#     embeddings = generate_embeddings(chunks, "RETRIEVAL_DOCUMENT")
#     vectors_to_upsert = [{"id": f"{document_id}_chunk_{i}", "values": emb, "metadata": {"text": chunk}} for i, (chunk, emb) in enumerate(zip(chunks, embeddings))]
#     for i in range(0, len(vectors_to_upsert), 100):
#         index.upsert(vectors=vectors_to_upsert[i:i + 100])
#     print(f"Successfully upserted {len(vectors_to_upsert)} vectors.")

# def query_pinecone(index, query_embedding: List[float], top_k: int) -> List[str]:
#     results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
#     return [match.metadata['text'] for match in results.matches]

# def hybrid_query(index, all_chunks: List[str], question: str, question_embedding: List[float], top_k: int) -> List[str]:
#     semantic_results = query_pinecone(index, question_embedding, top_k=3)
#     question_keywords = set(question.lower().split())
#     keyword_results = [chunk for chunk in all_chunks if any(keyword in chunk.lower() for keyword in question_keywords)]
#     combined_results = semantic_results + keyword_results[:4]
#     return list(dict.fromkeys(combined_results))

# # --- 6. LLM Answer Generation ---
# async def generate_multiple_queries(question: str) -> List[str]:
#     """Uses an LLM to generate variations of the user's question."""
#     llm_model = genai.GenerativeModel('gemini-1.5-flash-latest')
#     prompt = f"""You are a helpful assistant. Your task is to generate exaclty 3 different versions of a given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, you help the user overcome some of the limitations of distance-based similarity search. Provide these alternative questions separated by newlines.

#     Original question: {question}

#     Rephrased questions:"""
#     response = await llm_model.generate_content_async(prompt)
#     return [question] + [q for q in response.text.split('\n') if q]

# async def get_answer_for_question(question: str, all_chunks: List[str], semaphore: asyncio.Semaphore):
#     async with semaphore:
#         print(f"Processing question: '{question[:30]}...'")
        
#         # 1. Multi-Query Retrieval
#         generated_queries = await generate_multiple_queries(question)
#         print(f"Generated {len(generated_queries)} queries for retrieval.")
        
#         query_embeddings = generate_embeddings(generated_queries, "RETRIEVAL_QUERY")
        
#         all_retrieved_chunks = []
#         for i, query in enumerate(generated_queries):
#             chunks = hybrid_query(pinecone_index, all_chunks, query, query_embeddings[i], top_k=3)
#             all_retrieved_chunks.extend(chunks)
        
#         context_chunks = list(dict.fromkeys(all_retrieved_chunks))
#         context_str = "\n---\n".join(context_chunks)
        
#         # 2. Final Answer Generation
#         llm_model = genai.GenerativeModel('gemini-2.0-flash-lite')
#         prompt = f"""You are a precise and meticulous insurance policy analyst. Your task is to answer any user question based *exclusively* on the provided context.
#         **Instructions:**
#         1. Analyze the User's Question to understand the specific information needed.
#         2. Scan the Context to find all relevant clauses, definitions, and limits.
#         3. Think Step-by-Step: First, state the rule from the policy. Next, apply it to the scenario. Finally, provide a clear final answer.
#         4. If the information is not in the context, respond with the exact phrase: "The answer is not available in the provided document."
#         **Context:**
#         ---
#         {context_str}
#         ---
#         **Question:** {question}
#         **Answer:**"""
        
#         response = await llm_model.generate_content_async(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
#         final_answer = response.text.strip()
#         print(f"Final Answer Generated: {final_answer}")
#         return final_answer

# # --- 7. FastAPI Startup & Main Endpoint ---
# @app.on_event("startup")
# async def startup_event():
#     global pinecone_index
#     print("FastAPI application starting up...")
#     if PINECONE_INDEX_NAME not in pinecone_client.list_indexes().names():
#         pinecone_client.create_index(name=PINECONE_INDEX_NAME, dimension=EMBEDDING_DIM, metric="cosine", spec=ServerlessSpec(cloud='aws', region='us-east-1'))
#     pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME)
#     print(f"Pinecone index '{PINECONE_INDEX_NAME}' is ready.")

# @app.post("/api/v1/hackrx/run", response_model=AnswerResponse, dependencies=[Depends(verify_token)])
# async def run_submission(request: RunRequest):
#     start_time = time.time()
#     print(f"Received request for document: {request.documents}")
#     session_document_id = str(uuid.uuid4())
    
#     if request.documents in document_cache:
#         print("Found document text in cache.")
#         document_text = document_cache[request.documents]
#     else:
#         print("Document not in cache. Processing now.")
#         try:
#             response = requests.get(request.documents)
#             response.raise_for_status()
#             document_text = extract_text_from_pdf_stream(BytesIO(response.content))
#             document_cache[request.documents] = document_text
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

#     document_chunks = chunk_text(document_text)
#     upsert_chunks_to_pinecone(pinecone_index, document_chunks, session_document_id)

#     semaphore = asyncio.Semaphore(5)
#     tasks = [get_answer_for_question(q, document_chunks, semaphore) for q in request.questions]
#     results = await asyncio.gather(*tasks)

#     pinecone_index.delete(filter={"document_id": session_document_id})
#     print(f"--- Request completed in {time.time() - start_time:.2f} seconds ---")
#     return AnswerResponse(answers=results)
