import os
import uuid
import requests
from io import BytesIO
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict

from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration & Global Clients ---
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

# Initialize Pinecone and cache
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = None 
document_cache = {}

# --- FastAPI App Initialization ---
app = FastAPI(title="HackRx 6.0 Query-Retrieval System")

# --- Pydantic Models for Request/Response ---
class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

# --- Security Dependency ---
security = HTTPBearer()
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
    return True

# --- Document Processing Functions ---
# *******************************1 - Sequential Processing*******************************
# def extract_text_from_pdf_stream(file_stream: BytesIO) -> str:
#     """
#     Extracts text from a PDF file stream using sequential OCR to save memory.
#     """
#     text = ""
#     try:
#         images = convert_from_bytes(file_stream.read())
#         print(f"PDF converted to {len(images)} page(s) for OCR processing.")

#         # Process one page at a time to conserve memory
#         for i, image in enumerate(images):
#             print(f"Performing OCR on page {i+1}...")
#             text += pytesseract.image_to_string(image) or ""
        
#         print("OCR text extraction complete.")
#     except Exception as e:
#         print(f"Error extracting text via OCR: {e}")
#         raise ValueError(f"Failed to extract text from PDF using OCR: {e}")
        
#     return text

# *******************************2 - Batched Parallel Processing*******************************

# def extract_text_from_pdf_stream(file_stream: BytesIO) -> str:
#     """
#     Extracts text using batched parallel OCR to balance speed and memory usage.
#     """
#     all_texts = []
#     try:
#         images = convert_from_bytes(file_stream.read())
#         print(f"PDF converted to {len(images)} page(s) for OCR processing.")

#         # Process pages in small batches to control memory and CPU time
#         batch_size = 4 

#         def ocr_page(image):
#             return pytesseract.image_to_string(image) or ""

#         with ThreadPoolExecutor() as executor:
#             # Loop through the images in batches
#             for i in range(0, len(images), batch_size):
#                 batch = images[i:i + batch_size]
#                 print(f"Processing batch starting at page {i+1}...")
                
#                 # Run OCR on the current batch in parallel
#                 texts = list(executor.map(ocr_page, batch))
#                 all_texts.extend(texts)
        
#         print("Batched parallel OCR text extraction complete.")
#         return "".join(all_texts)
        
#     except Exception as e:
#         print(f"Error extracting text via OCR: {e}")
#         raise ValueError(f"Failed to extract text from PDF using OCR: {e}")

# *******************************3 - Complete Parallel Processing*******************************
def extract_text_from_pdf_stream(file_stream: BytesIO) -> str:
    """
    Extracts text from a PDF file stream using fully parallelized OCR for maximum speed.
    """
    try:
        images = convert_from_bytes(file_stream.read())
        print(f"PDF converted to {len(images)} page(s) for OCR processing.")

        def ocr_page(image):
            # This function will be run in parallel for each page
            return pytesseract.image_to_string(image) or ""

        # Use a ThreadPoolExecutor to run OCR on all pages concurrently
        with ThreadPoolExecutor() as executor:
            texts = list(executor.map(ocr_page, images))
        
        print("Parallel OCR text extraction complete.")
        return "".join(texts)
        
    except Exception as e:
        print(f"Error extracting text via OCR: {e}")
        raise ValueError(f"Failed to extract text from PDF using OCR: {e}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Splits text into chunks with overlap."""
    if not text:
        return []
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]
    
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
    return chunks

# --- Embedding and Vector Store Functions ---
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
        batch = vectors_to_upsert[i:i + 100]
        index.upsert(vectors=batch)
    print(f"Successfully upserted {len(vectors_to_upsert)} vectors.")

def query_pinecone(index, query_embedding: List[float], top_k: int) -> List[str]:
    """Performs a semantic search in Pinecone."""
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [match.metadata['text'] for match in results.matches]

def hybrid_query(index, all_chunks: List[str], question: str, question_embedding: List[float], top_k: int) -> List[str]:
    """Combines semantic search with keyword search for improved accuracy."""
    semantic_results = query_pinecone(index, question_embedding, top_k=top_k)
    
    question_keywords = set(question.lower().split())
    keyword_results = [chunk for chunk in all_chunks if any(keyword in chunk.lower() for keyword in question_keywords)]
    
    combined_results = semantic_results + keyword_results
    unique_results = list(dict.fromkeys(combined_results))
    print(f"Found {len(unique_results)} unique chunks with hybrid search.")
    return unique_results

# --- LLM Processing Function (RAG) ---
async def get_answer_for_question(question: str, question_embedding: List[float], all_chunks: List[str], semaphore: asyncio.Semaphore):
    """Asynchronously gets a single answer for a single question using a hybrid search context."""
    async with semaphore:
        print(f"Processing question: '{question[:30]}...'")
        
        context_chunks = hybrid_query(pinecone_index, all_chunks, question, question_embedding, top_k=5)
        context_str = "\n---\n".join(context_chunks)
        
        llm_model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""You are a precise and meticulous insurance policy analyst. Your task is to answer questions based *exclusively* on the provided context.
        **Instructions:**
        1. Carefully read the provided context.
        2. Answer the user's question using only the information found in the context.
        3. If the answer is a specific number, amount, or time period, quote it exactly.
        4. If the information is not available in the context, you MUST respond with the exact phrase: "The answer is not available in the provided document."
        5. Do not invent, assume, or infer any information not explicitly stated in the context.

        **Context:**
        ---
        {context_str}
        ---
        **Question:** {question}
        **Answer:**"""
        
        response = await llm_model.generate_content_async(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.1)
        )
        return response.text.strip()

# --- FastAPI Startup & Main Endpoint ---
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

@app.post("/hackrx/run", response_model=AnswerResponse, dependencies=[Depends(verify_token)])
async def run_submission(request: RunRequest):
    """Main endpoint to process documents and answer questions."""
    start_time = time.time()
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

# --- How to Run Locally ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
