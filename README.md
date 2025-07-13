# ChromaDB + Streamlit PDF Question Answering System

This project implements a retrieval-augmented question answering (QA) system that ingests PDF documents, embeds the content using a local transformer model, stores the vectors in ChromaDB, and uses OpenAI GPT-4o (Azure-hosted) to answer natural language questions through a Streamlit interface.

## Features

- Embeds PDF content locally using `all-MiniLM-L6-v2` from SentenceTransformers
- Stores vectors in a persistent ChromaDB collection
- Provides a question-answering interface via GPT-4o (Azure-hosted)
- Displays response time and query performance metrics
- Supports offline embedding (no API key required for embeddings)

## Project Structure

