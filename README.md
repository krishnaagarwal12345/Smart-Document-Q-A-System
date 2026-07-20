# 📄 PDF Chat — Smart Document Q&A System

## Overview
A production-level RAG (Retrieval Augmented Generation) 
system that allows you to upload ANY PDF and chat with it 
using AI. Ask questions and get accurate answers directly 
from your document!

## 🎯 How It Works
Upload PDF → Extract Text → Create Chunks
↓
Convert to Embeddings → Store in FAISS
↓
Ask Question → Semantic Search → Find Relevant Sections
↓
RAG Prompt → Llama AI → Accurate Answer + Sources

## ✨ Features

✅ Upload any PDF via drag & drop or click
✅ Automatic text extraction from all pages
✅ Smart chunking with overlap (no context lost)
✅ Sentence transformer embeddings (384 dimensions)
✅ FAISS vector database for fast semantic search
✅ RAG pipeline — answers only from YOUR document
✅ Source sections displayed for transparency
✅ Response time shown for each answer
✅ Clear PDF and upload new one anytime
✅ Professional UI with typing indicator
✅ Error handling at all layers

## 🛠️ Tech Stack

### AI Backend (Python)
- **FastAPI** — REST API framework
- **PyMuPDF (fitz)** — PDF text extraction
- **Sentence Transformers** — text embeddings
- **FAISS** — vector database for similarity search
- **Groq API** — Llama 3.3 70B language model
- **Uvicorn** — ASGI server

### Frontend
- **HTML5** — structure
- **CSS3** — professional UI design
- **JavaScript** — fetch API, drag & drop

## 📁 Project Structure

pdf-rag-system/
├── SmartQandA_CHATBot.py ← Python FastAPI backend
├── pdf_chat.html ← Frontend UI
├── requirements.txt ← Python dependencies
└── README.md

## 🚀 How to Run

### Step 1 — Install dependencies
```bash
pip install fastapi uvicorn groq sentence-transformers faiss-cpu numpy pymupdf python-multipart
```

### Step 2 — Set API key
```bash
# Get free API key from console.groq.com
# Replace in SmartQandA_CHATBot.py:
GROQ_API_KEY = "your_groq_api_key_here"
```

### Step 3 — Start backend
```bash
python -m uvicorn SmartQandA_CHATBot:app --reload --port 8001
```

### Step 4 — Open frontend
Open pdf_chat.html in your browser
OR
Go to http://localhost:8001/chat

### Step 5 — Start chatting!
Upload any PDF
Wait for processing
Ask any question
Get accurate answers!

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Home — API status |
| GET | /health | Health check |
| POST | /upload | Upload PDF file |
| POST | /ask | Ask question about PDF |
| DELETE | /clear | Clear current PDF |
| GET | /info | Get PDF information |
| GET | /docs | Swagger UI |

## 📊 RAG Pipeline Details

### 1. PDF Processing
```python
# Extract text from all pages
doc = fitz.open(stream=pdf_bytes, filetype="pdf")
for page in doc:
    text += page.get_text()
```

### 2. Text Chunking
```python
# 500 words per chunk, 50 word overlap
chunk_size = 500
overlap = 50
```

### 3. Embedding Creation
```python
# 384 dimensional vectors
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)
```

### 4. FAISS Storage
```python
# L2 distance similarity search
index = faiss.IndexFlatL2(384)
index.add(embeddings)
```

### 5. Semantic Search
```python
# Find top 3 most relevant chunks
distances, indices = index.search(question_embedding, k=3)
```

### 6. Answer Generation
```python
# RAG prompt — answers only from PDF context
prompt = f"""Answer using ONLY this context:
{relevant_chunks}
Question: {question}"""
```

## 💡 Example Usage

Upload: company_policy.pdf

Question: "How many leave days do employees get?"
Answer: "According to the document, employees receive
20 days of annual leave per year..."
Sources: [Section 1: The company offers 20 days...]

Question: "What are the office hours?"
Answer: "Office hours are 9 AM to 6 PM Monday to Friday..."
Sources: [Section 2: Office hours are 9 AM...]

## 🎯 Use Cases

✅ Chat with research papers
✅ Query legal documents
✅ Analyze company policies
✅ Study textbooks
✅ Extract information from reports
✅ Interview preparation from study material

## ⚙️ Configuration

```python
# In SmartQandA_CHATBot.py
GROQ_API_KEY = "your_key"      # Groq API key
MODEL = "llama-3.3-70b-versatile"  # AI model
chunk_size = 500               # Words per chunk
overlap = 50                   # Overlap between chunks
k = 3                          # Top chunks to retrieve
```

## 📦 Requirements

fastapi
uvicorn[standard]
groq
sentence-transformers
faiss-cpu
numpy
pymupdf
python-multipart
pydantic


## 🔒 Important Notes

⚠️ Never commit your API key to GitHub!
⚠️ Use environment variables for API keys
⚠️ PDF must contain text (not scanned images)
⚠️ Free Groq API has rate limits


## 🌟 Key Concepts Implemented

| Concept | Implementation |
|---------|----------------|
| RAG | Retrieval Augmented Generation pipeline |
| Embeddings | Sentence Transformers (384 dimensions) |
| Vector Search | FAISS IndexFlatL2 |
| PDF Processing | PyMuPDF text extraction |
| Chunking | Fixed size with overlap |
| API | FastAPI REST endpoints |
| Frontend | Vanilla HTML/CSS/JS |

## 📈 Performance

PDF Processing: ~5-10 seconds per MB
Embedding creation: ~2-3 seconds per 100 chunks
Search time: <100ms (FAISS optimized)
Answer generation: ~2-5 seconds (Groq API)


## 🎓 Learning Outcomes

Building this project taught:
- Complete RAG pipeline from scratch
- PDF text extraction with PyMuPDF
- Semantic search with FAISS
- FastAPI file upload handling
- Async Python programming
- Frontend + Backend integration

## 👨‍💻 Created By

**Krishna Agarwal**
Final Year CS Student | AI/ML Developer
Java + Python + GenAI + Full Stack

🔗 GitHub: github.com/krishnaagarwal12345
🔗 LinkedIn: linkedin.com/in/krishna-agarwal-922ab0375



---
⭐ Star this repository if you found it helpful!
