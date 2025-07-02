# RAG Document Assistant

## Overview

This is a Retrieval-Augmented Generation (RAG) application built with Streamlit that allows users to upload documents and interact with them through an AI-powered chat interface. The system extracts text from various document formats, creates vector embeddings, and uses OpenAI's GPT models to answer questions based on the uploaded content.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

- **Frontend**: Streamlit web interface for document upload and chat interaction
- **Document Processing**: Text extraction from PDF, TXT, and DOCX files
- **Vector Storage**: Document embeddings using OpenAI's embedding models
- **RAG Engine**: Question-answering system using retrieved context and GPT-4o

## Key Components

### 1. Main Application (`app.py`)
- **Purpose**: Streamlit web interface and application orchestration
- **Features**: Document upload, chat interface, session state management
- **Dependencies**: Requires OpenAI API key environment variable

### 2. Document Processor (`document_processor.py`)
- **Purpose**: Extract and chunk text from various document formats
- **Supported Formats**: PDF, TXT, DOCX
- **Chunking Strategy**: Fixed-size chunks (1000 chars) with overlap (200 chars)
- **Libraries**: PyPDF2 for PDFs, python-docx for Word documents

### 3. Vector Store (`vector_store.py`)
- **Purpose**: Generate and manage document embeddings for similarity search
- **Embedding Model**: OpenAI's "text-embedding-3-small"
- **Storage**: Pickle-based persistence to disk
- **Search**: Cosine similarity for document retrieval

### 4. RAG Engine (`rag_engine.py`)
- **Purpose**: Orchestrate retrieval and generation for question-answering
- **Model**: GPT-4o (latest OpenAI model as of May 2024)
- **Retrieval**: Top-k similarity search (default: 5 chunks)
- **Context Preparation**: Combines relevant document chunks with source attribution

## Data Flow

1. **Document Upload**: User uploads PDF/TXT/DOCX files through Streamlit interface
2. **Text Extraction**: DocumentProcessor extracts text based on file type
3. **Chunking**: Text is split into overlapping chunks for better retrieval
4. **Embedding Generation**: VectorStore creates embeddings using OpenAI API
5. **Storage**: Documents and embeddings are persisted to disk
6. **Query Processing**: User questions trigger similarity search in vector store
7. **Context Retrieval**: Most relevant document chunks are retrieved
8. **Response Generation**: RAG engine combines context with question for GPT-4o
9. **Answer Display**: Generated response is shown in chat interface

## External Dependencies

### Required APIs
- **OpenAI API**: For embeddings (text-embedding-3-small) and chat completion (GPT-4o)
- **API Key**: Must be set as OPENAI_API_KEY environment variable

### Python Libraries
- **streamlit**: Web application framework
- **openai**: Official OpenAI Python client
- **PyPDF2**: PDF text extraction
- **python-docx**: Word document processing
- **numpy**: Numerical operations for embeddings
- **pickle**: Data serialization for vector storage

## Deployment Strategy

### Current Implementation
- **Platform**: Designed for Replit deployment
- **Storage**: Local file system with pickle serialization
- **Session Management**: Streamlit session state for user context
- **Environment**: Requires Python 3.7+ with pip package management

### Scalability Considerations
- Vector store uses in-memory storage with disk persistence
- Single-user session state (no multi-user support)
- API rate limits may affect batch processing of large documents

## Changelog
- July 02, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.