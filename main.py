# ==============================================================================
# Final main.py for HackRx 6.0 - Intelligent AI Agent Architecture
# ==============================================================================

import os
import uuid
import requests
import asyncio
import time
import json
import re
from io import BytesIO
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

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
from pypdf import PdfReader


# --- 1. Configuration & Clients ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

genai.configure(api_key=GEMINI_API_KEY)

EMBEDDING_MODEL = "models/embedding-001"
EMBEDDING_DIM = 768

pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = None 
document_cache = {}


# --- 2. FastAPI App & Models ---
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
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
    return True


# --- 4. Document Processing (AI Agent Workflow) ---

def get_relevant_pages(file_stream: BytesIO, question: str) -> List[int]:
    """
    Planner Agent: Analyzes the Table of Contents to decide which pages are relevant,
    avoiding the need to process the entire document.
    """
    print(f"Planner Agent: Determining relevant pages for question: '{question[:30]}...'")
    toc = []
    page_count = 0
    try:
        pdf_reader = PdfReader(file_stream)
        page_count = len(pdf_reader.pages)
        if hasattr(pdf_reader, 'outline') and pdf_reader.outline:
            for item in pdf_reader.outline:
                if hasattr(item, 'title') and hasattr(item, 'page'):
                    # Get page number (outline can be indirect, resolve it)
                    page_obj = item.page
                    page_number = pdf_reader.get_page_number(page_obj) + 1
                    toc.append(f"Page {page_number}: {item.title}")
    except Exception as e:
        print(f"Could not automatically extract Table of Contents. Error: {e}")
        toc = ["No table of contents available."]

    toc_str = "\n".join(toc)
    llm_model = genai.GenerativeModel('gemini-2.0-flash-lite')
    
    prompt = f"""You are a document routing agent. Your task is to identify the most relevant page numbers from a document to answer a user's question, based on the document's Table of Contents. The document has {page_count} pages.

    Return a Python list of the most likely page numbers. For example: [10, 11, 25].
    If the table of contents is not helpful or unavailable, make a reasonable guess based on the question's keywords (e.g., questions about 'exclusions' are often near the end of a policy). Return no more than 15 pages.

    Table of Contents:
    ---
    {toc_str}
    ---

    Question: {question}

    Relevant Page Numbers (as a Python list):
    """
    try:
        response = llm_model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
        match = re.search(r'\[([\d,\s]*)\]', response.text)
        if match:
            page_numbers = sorted(list(set([int(p.strip()) for p in match.group(1).split(',') if p.strip()])))
            print(f"Planner Agent identified relevant pages: {page_numbers}")
            return page_numbers if page_numbers else list(range(1, min(page_count, 11)))
    except Exception as e:
        print(f"Planner Agent failed, defaulting to first 10 pages. Error: {e}")
    
    # Fallback if the LLM fails or no ToC is available
    return list(range(1, min(page_count + 1, 11)))

def extract_text_from_pdf_stream(file_stream: BytesIO, question: str) -> str:
    """
    Intelligently extracts text from only the most relevant pages of a PDF.
    """
    stream_for_planner = BytesIO(file_stream.getvalue())
    relevant_pages = get_relevant_pages(stream_for_planner, question)
    # Convert 1-based page numbers to 0-based indices for pypdf
    relevant_page_indices = [p - 1 for p in relevant_pages] 

    text = ""
    stream_for_extraction = BytesIO(file_stream.getvalue())
    try:
        print(f"Attempting fast direct text extraction on {len(relevant_pages)} relevant pages...")
        pdf_reader = PdfReader(stream_for_extraction)
        for i in relevant_page_indices:
            if i < len(pdf_reader.pages):
                page_text = pdf_reader.pages[i].extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Pypdf extraction failed: {e}")
        text = ""

    if len(text.strip()) > 100:
        print("Direct text extraction successful from relevant pages.")
        return text

    print("Direct extraction failed. Falling back to OCR on relevant pages...")
    return extract_text_with_ocr(BytesIO(file_stream.getvalue()), relevant_pages)

def extract_text_with_ocr(file_stream: BytesIO, pages_to_process: List[int]) -> str:
    """Performs OCR only on a specific subset of pages."""
    try:
        # The 'user_pages' parameter is the key to only processing specific pages
        images = convert_from_bytes(file_stream, user_pages=pages_to_process)
        print(f"Extracted {len(images)} page(s) for OCR processing.")

        def ocr_page(image):
            return pytesseract.image_to_string(image) or ""

        with ThreadPoolExecutor() as executor:
            texts = list(executor.map(ocr_page, images))
        
        return "".join(texts)
    except Exception as e:
        raise ValueError(f"Failed to perform OCR on selected pages: {e}")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    if not text: return []
    words = text.split()
    if len(words) <= chunk_size: return [" ".join(words)]
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


# --- 5. Retrieval & Vector DB Functions ---
def generate_embeddings(texts: List[str], task_type: str) -> List[List[float]]:
    if not texts: return []
    # Clean texts before sending to embedding model
    cleaned_texts = [text.replace('\n', ' ').strip() for text in texts if text.strip()]
    if not cleaned_texts: return []
    response = genai.embed_content(model=EMBEDDING_MODEL, content=cleaned_texts, task_type=task_type)
    return response['embedding']

def upsert_chunks_to_pinecone(index, chunks: List[str], document_id: str):
    if not chunks: return
    embeddings = generate_embeddings(chunks, "RETRIEVAL_DOCUMENT")
    vectors_to_upsert = [{"id": f"{document_id}_chunk_{i}", "values": emb, "metadata": {"text": chunk, "document_id": document_id}} for i, (chunk, emb) in enumerate(zip(chunks, embeddings))]
    for i in range(0, len(vectors_to_upsert), 100):
        index.upsert(vectors=vectors_to_upsert[i:i + 100])
    print(f"Successfully upserted {len(vectors_to_upsert)} vectors.")

def query_pinecone(index, query_embedding: List[float], top_k: int) -> List[str]:
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [match.metadata['text'] for match in results.matches]

def hybrid_query(index, all_chunks: List[str], question: str, question_embedding: List[float], top_k: int) -> List[str]:
    semantic_results = query_pinecone(index, question_embedding, top_k=3)
    question_keywords = set(question.lower().split())
    keyword_results = [chunk for chunk in all_chunks if any(keyword in chunk.lower() for keyword in question_keywords)]
    combined_results = semantic_results + keyword_results[:4] # Token efficiency control
    return list(dict.fromkeys(combined_results))


# --- 6. AI Agent Functions ---
async def generate_multiple_queries(question: str) -> List[str]:
    """Uses an LLM to generate variations of the user's question for better retrieval."""
    llm_model = genai.GenerativeModel('gemini-2.0-flash-lite')
    prompt = f"""You are a helpful assistant. Your task is to generate 3 different versions of a given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, you help the user overcome some of the limitations of distance-based similarity search. Provide these alternative questions separated by newlines.

    Original question: {question}

    Rephrased questions:"""
    try:
        response = await llm_model.generate_content_async(prompt)
        return [question] + [q for q in response.text.split('\n') if q and len(q) > 5]
    except Exception:
        return [question]

async def get_answer_for_question(question: str, all_chunks: List[str], semaphore: asyncio.Semaphore):
    """Orchestrates the multi-step agentic process for a single question."""
    async with semaphore:
        print(f"Answering Agent: Processing question: '{question[:30]}...'")
        
        # Step 1: Multi-Query Generation for better retrieval
        generated_queries = await generate_multiple_queries(question)
        
        # Step 2: Multi-Query Retrieval
        query_embeddings = generate_embeddings(generated_queries, "RETRIEVAL_QUERY")
        
        all_retrieved_chunks = []
        for i, query in enumerate(generated_queries):
            if i < len(query_embeddings): # Ensure we have an embedding for the query
                chunks = hybrid_query(pinecone_index, all_chunks, query, query_embeddings[i], top_k=3)
                all_retrieved_chunks.extend(chunks)
        
        context_chunks = list(dict.fromkeys(all_retrieved_chunks))
        context_str = "\n---\n".join(context_chunks)
        
        # Step 3: Final Answer Generation
        llm_model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""You are a precise insurance policy analyst. Your task is to provide a clear and concise answer to the user's question based *exclusively* on the provided context.
        **Instructions:**
        1. Analyze the Question and Context to find the answer.
        2. Think Step-by-Step: State the relevant policy rules, apply them to the scenario, show calculations if needed, and form a definitive conclusion.
        3. Provide a direct, self-contained, professional answer. For yes/no questions, start with "Yes" or "No".
        4. If the information is not in the context, respond with: "The answer is not available in the provided document."
        **Context:**
        ---
        {context_str}
        ---
        **Question:** {question}
        **Answer:**"""
        
        response = await llm_model.generate_content_async(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
        final_answer = response.text.strip()
        print(f"Answering Agent: Final Answer Generated: {final_answer}")
        return final_answer

# --- 7. FastAPI Startup & Main Endpoint ---
@app.on_event("startup")
async def startup_event():
    global pinecone_index
    print("FastAPI application starting up...")
    if PINECONE_INDEX_NAME not in pinecone_client.list_indexes().names():
        pinecone_client.create_index(name=PINECONE_INDEX_NAME, dimension=EMBEDDING_DIM, metric="cosine", spec=ServerlessSpec(cloud='aws', region='us-east-1'))
    pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME)
    print(f"Pinecone index '{PINECONE_INDEX_NAME}' is ready.")

@app.post("/api/v1/hackrx/run", response_model=AnswerResponse, dependencies=[Depends(verify_token)])
async def run_submission(request: RunRequest):
    start_time = time.time()
    print(f"Received request for document: {request.documents}")
    
    try:
        # Download the document once and cache it
        if request.documents in document_cache:
            print("Found document bytes in cache.")
            document_stream = BytesIO(document_cache[request.documents])
        else:
            print("Document not in cache. Downloading now.")
            response = requests.get(request.documents)
            response.raise_for_status()
            document_cache[request.documents] = response.content
            document_stream = BytesIO(response.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download or cache document: {e}")

    # The agentic workflow is now per-question, so we create a list of tasks
    semaphore = asyncio.Semaphore(3) # Control concurrency for the entire process
    tasks = [process_single_question(q, document_stream, semaphore) for q in request.questions]
    results = await asyncio.gather(*tasks)

    print(f"--- Full request completed in {time.time() - start_time:.2f} seconds ---")
    return AnswerResponse(answers=results)

async def process_single_question(question: str, document_stream: BytesIO, semaphore: asyncio.Semaphore):
    """New helper function to orchestrate the agentic workflow for one question."""
    session_id = str(uuid.uuid4())
    document_stream.seek(0)
    
    # 1. Smart Text Extraction (from relevant pages only)
    document_text = extract_text_from_pdf_stream(document_stream, question)
    if not document_text.strip():
        return "Could not extract any relevant text from the document for this question."
        
    # 2. Chunking and Upserting only the relevant text
    document_chunks = chunk_text(document_text)
    upsert_chunks_to_pinecone(pinecone_index, document_chunks, session_id)
    
    # 3. Answering
    answer = await get_answer_for_question(question, document_chunks, semaphore)
    
    # 4. Cleanup
    pinecone_index.delete(filter={"document_id": session_id})
    
    return answer
