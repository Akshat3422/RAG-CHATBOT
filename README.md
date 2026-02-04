# 🤖 RAG Chatbot – Document-Based Conversational AI

The **RAG Chatbot** is a document-centric conversational application built using the **Retrieval-Augmented Generation (RAG)** architecture. The project enables users to upload documents (PDFs) and interact with their content through natural language queries. By combining semantic retrieval with large language models, the system generates accurate, context-aware answers grounded in the uploaded documents.

---

## 📌 Features

- Upload and ingest PDF documents
- Semantic search using vector embeddings
- Context-aware responses using RAG
- FastAPI-based backend APIs
- Streamlit-based interactive frontend
- Namespace-based document isolation
- Secure handling of API keys using environment variables

---

## 🏗️ Architecture Overview

1. **Document Ingestion**
   - PDF documents are uploaded via the frontend
   - Text is extracted, chunked, and embedded
   - Embeddings are stored in a vector database

2. **Query Processing**
   - User queries are embedded
   - Relevant document chunks are retrieved using similarity search
   - Retrieved context is passed to an LLM for answer generation

3. **Response Generation**
   - The LLM generates responses grounded in retrieved documents
   - Reduces hallucinations and improves factual accuracy

---

## 🛠️ Tech Stack

- **Backend:** FastAPI
- **Frontend:** Streamlit
- **LLM Integration:** Groq / Google GenAI (via LangChain)
- **Embeddings:** Sentence Transformers
- **Vector Database:** Pinecone
- **Document Parsing:** PyPDF
- **Language:** Python

---
