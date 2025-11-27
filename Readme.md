## Document Q&A Assistant (RAG-Based Mini Project)

### Overview

This project implements a lightweight Retrieval-Augmented Generation (RAG) system that allows users to upload documents and ask questions strictly based on the content inside those documents. 
The system extracts text, splits it into semantic chunks, creates embeddings, stores them in a vector store, retrieves the most relevant chunks, and uses an LLM to generate grounded responses.

The assistant never hallucinates; if the answer is not present in the provided documents, it explicitly responds:

“I don’t have this information in the provided documents.”

The application is built using Streamlit + LangChain + Ollama and runs entirely locally (or via a hosted Ollama endpoint).

## Libraries Used
### Core Libraries
LangChain
Document Loaders (PDF, TXT, CSV, DOCX)
RecursiveCharacterTextSplitter
Vector Search / RAG utilities
LangChain-Ollama
Vector Store
InMemoryVectorStore (lightweight, fast for demo use)

Web Interface
Streamlit for UI
Custom HTML/CSS for UX enhancements

## RAG Workflow

Document Upload
Users upload files (PDF, TXT, CSV, DOCX).

Text Extraction & Chunking
Documents are split into ~400-token chunks with overlap.

Embedding Generation
Each chunk is embedded using mxbai-embed-large.

Vector Search
At query time, top-k relevant chunks are retrieved using cosine similarity.

Similarity threshold ≥ 0.30 is required

LLM Response (Context-restricted)
The LLM is instructed to answer only using retrieved chunks.

Out-of-Scope Handling
If no relevant chunks pass the threshold, system returns:
“I don’t have this information in the provided documents.”

## Features

Multi-format document ingestion
Real-time Q&A over uploaded documents
Similarity-based retrieval with thresholding
No hallucination due to strict prompt and gating
Reference source display (chunk ID + file)
Clean and responsive Streamlit UI

RAG Workflow

Docum
