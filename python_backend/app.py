from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from groq import Groq
import fitz  # PyMuPDF
import faiss
import numpy as np
import os
import time
from sentence_transformers import SentenceTransformer

# ============================================
# CONFIGURATION
# ============================================
GROQ_API_KEY =os.environ.get("GROQ_API_KEY")

MODEL = "llama-3.3-70b-versatile"

# ============================================
# INITIALIZE
# ============================================
print("Starting PDF RAG System...")
groq_client = Groq(api_key=GROQ_API_KEY)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Models loaded! ✅")

# ============================================
# FASTAPI APP
# ============================================
app = FastAPI(
    title="PDF RAG System",
    description="Upload any PDF and chat with it!",
    version="1.0.0"
)
# Add after app = FastAPI(...)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add this new endpoint
@app.get("/chat")
def chat_page():
    from fastapi.responses import FileResponse
    return FileResponse("static/pdf_chat.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# GLOBAL STORAGE
# ============================================
# Stores current PDF data in memory
pdf_store = {
    "filename": None,
    "chunks": [],
    "index": None,
    "total_pages": 0,
    "total_chars": 0
}

# ============================================
# HELPER FUNCTIONS
# ============================================
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all text from PDF bytes"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text, len(doc)

def chunk_text(text: str,
               chunk_size: int = 500,
               overlap: int = 50) -> list:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def build_faiss_index(chunks: list):
    """Build FAISS index from chunks"""
    embeddings = embed_model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

def search_relevant_chunks(question: str,
                           chunks: list,
                           index,
                           k: int = 3) -> list:
    """Find most relevant chunks for question"""
    q_embed = embed_model.encode([question])
    distances, indices = index.search(
        np.array(q_embed), k
    )
    return [chunks[indices[0][i]] for i in range(k)]

# ============================================
# REQUEST / RESPONSE MODELS
# ============================================
class QuestionRequest(BaseModel):
    question: str

class UploadResponse(BaseModel):
    message: str
    filename: str
    total_pages: int
    total_chunks: int
    total_chars: int

class AnswerResponse(BaseModel):
    question: str
    answer: str
    relevant_chunks: list
    model: str
    time_taken: float

# ============================================
# ENDPOINTS
# ============================================
@app.get("/")
def home():
    return {
        "message": "PDF RAG System Running! 📄🤖",
        "status": "ready",
        "pdf_loaded": pdf_store["filename"] is not None,
        "current_pdf": pdf_store["filename"]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "pdf_loaded": pdf_store["filename"] is not None,
        "current_pdf": pdf_store["filename"],
        "total_chunks": len(pdf_store["chunks"])
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file and process it"""

    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed!"
        )

    print(f"Processing PDF: {file.filename}")

    try:
        # Read PDF bytes
        pdf_bytes = await file.read()

        # Extract text
        print("Extracting text...")
        text, total_pages = extract_text_from_pdf(pdf_bytes)

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="PDF appears to be empty or scanned image — text extraction failed!"
            )

        # Chunk text
        print("Creating chunks...")
        chunks = chunk_text(text)
        print(f"Created {len(chunks)} chunks")

        # Build FAISS index
        print("Building FAISS index...")
        index = build_faiss_index(chunks)
        print("FAISS index ready!")

        # Store in memory
        pdf_store["filename"] = file.filename
        pdf_store["chunks"] = chunks
        pdf_store["index"] = index
        pdf_store["total_pages"] = total_pages
        pdf_store["total_chars"] = len(text)

        return UploadResponse(
            message=f"PDF processed successfully!",
            filename=file.filename,
            total_pages=total_pages,
            total_chunks=len(chunks),
            total_chars=len(text)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """Ask a question about the uploaded PDF"""

    # Check if PDF is loaded
    if pdf_store["index"] is None:
        raise HTTPException(
            status_code=400,
            detail="No PDF uploaded yet! Please upload a PDF first."
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty!"
        )

    start = time.time()

    try:
        # Find relevant chunks
        relevant = search_relevant_chunks(
            request.question,
            pdf_store["chunks"],
            pdf_store["index"]
        )

        # Build RAG prompt
        context = "\n\n".join([
            f"Section {i+1}:\n{chunk}"
            for i, chunk in enumerate(relevant)
        ])

        prompt = f"""You are a helpful assistant analyzing a PDF document.
Answer the question using ONLY the provided context from the PDF.
If the answer is not in the context, say "This information is not found in the uploaded PDF."

Context from PDF:
{context}

Question: {request.question}

Answer:"""

        # Call AI
        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        answer = response.choices[0].message.content
        time_taken = round(time.time() - start, 2)

        return AnswerResponse(
            question=request.question,
            answer=answer,
            relevant_chunks=relevant,
            model=MODEL,
            time_taken=time_taken
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer: {str(e)}"
        )

@app.delete("/clear")
def clear_pdf():
    """Clear the current PDF from memory"""
    pdf_store["filename"] = None
    pdf_store["chunks"] = []
    pdf_store["index"] = None
    pdf_store["total_pages"] = 0
    pdf_store["total_chars"] = 0
    return {"message": "PDF cleared successfully!"}

@app.get("/info")
def pdf_info():
    """Get information about currently loaded PDF"""
    if pdf_store["filename"] is None:
        return {"message": "No PDF loaded"}
    return {
        "filename": pdf_store["filename"],
        "total_pages": pdf_store["total_pages"],
        "total_chunks": len(pdf_store["chunks"]),
        "total_chars": pdf_store["total_chars"]
    }