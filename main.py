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
    semantic_results = query_pinecone(index, question_embedding, top_k=3) # Reduced to 3
    
    # 2. Perform keyword search
    question_keywords = set(question.lower().split())
    keyword_results = [chunk for chunk in all_chunks if any(keyword in chunk.lower() for keyword in question_keywords)]
    
    # 3. Combine results, limiting keyword results to prevent excessive token usage
    # We take the best 3 semantic results and add up to 5 relevant keyword results
    combined_results = semantic_results + keyword_results[:5] # Limit keyword results
    
    unique_results = list(dict.fromkeys(combined_results))
    print(f"Found {len(unique_results)} unique chunks with hybrid search.")
    return unique_results


# --- 6. LLM Answer Generation ---

async def get_answer_for_question(question: str, question_embedding: List[float], all_chunks: List[str], semaphore: asyncio.Semaphore):
    """Asynchronously gets a single answer for a single question using a hybrid search context."""
    async with semaphore:
        print(f"Processing question: '{question[:]}...'")
        
        context_chunks = hybrid_query(pinecone_index, all_chunks, question, question_embedding, top_k=5)
        context_str = "\n---\n".join(context_chunks)
        
        llm_model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        prompt = f"""You are a precise and meticulous insurance policy analyst. Your task is to answer any user question based *exclusively* on the provided context.

        **Instructions:**
        1.  **Analyze the User's Question:** First, identify the type of question being asked (e.g., direct fact, yes/no, or a scenario with specific details like names, ages, or dates).
        2.  **Scan the Context:** Thoroughly search the provided context for all relevant clauses, definitions, limits, waiting periods, and exclusions that relate to the question.
        3.  **Think Step-by-Step:**
            * **State the Rule:** Begin by quoting the exact rule, clause, or definition from the policy that applies to the question. If multiple clauses are relevant, state them all.
            * **Apply to Scenario (if applicable):** If the question is a scenario, explicitly apply the rules to the specific facts of the scenario. Show any necessary calculations.
            * **Formulate the Conclusion:** Based *only* on the rules and your analysis, form a definitive conclusion. For yes/no questions, the final answer must start with a clear "Yes" or "No".
        4.  **Provide the Final Answer:** Combine your reasoning into a clear, final answer.

        **Crucial Rule:** If the information to answer the question is not available in the context, you MUST respond with the exact phrase: "The answer is not available in the provided document." Do not invent, assume, or infer any information.

        **Context:**
        ---
        {context_str}
        ---

        **Question:** {question}

        **JSON Response:**
        ```json
        """
        
        response = await llm_model.generate_content_async(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.0)
        )
        
        final_answer = "Error: Could not parse LLM response." # Default error message
        try:
            # Clean up and parse the JSON response from the model
            json_text = response.text.replace("```json", "").replace("```", "").strip()
            parsed_response = json.loads(json_text)
            final_answer = parsed_response.get("answer", "Error: 'answer' key not found in LLM response.")
        except (json.JSONDecodeError, AttributeError):
            # If the model fails to return valid JSON, use its raw text as a fallback
            final_answer = response.text.strip()
        
        # Print the final answer to the server logs
        print(f"Final Answer Generated: {final_answer[:15]}")
        
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
