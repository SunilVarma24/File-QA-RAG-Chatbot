# File QA RAG Chatbot using Gemini 1.5 Pro, ChromaDB, and Streamlit

## Project Overview
This project implements a File Question Answering (QA) Chatbot using a Retrieval-Augmented Generation (RAG) approach. The chatbot allows users to upload custom documents, which are processed into vector embeddings and stored in ChromaDB, a vector database. When a query is input, the chatbot retrieves the most relevant information from the database using a similarity search and provides answers based on the context of the documents. The system is deployed as a web application using Streamlit, offering an easy-to-use interface for end users.

## Introduction
The File QA RAG Chatbot is designed to assist users in extracting relevant information from custom documents through natural language queries. The system combines the power of a large language model (Gemini 1.5 Pro) with vector-based retrieval using ChromaDB. This approach enables the chatbot to provide accurate and contextually relevant answers based on the contents of the uploaded documents, making it a valuable tool for document-based question-answering tasks.

## How It Works
1. **Document Upload**: Users upload a document to the chatbot.
2. **Document Processing**: The document is split into small chunks, and each chunk is converted into a vector embedding using an embedding model.
3. **Vector Storage**: The embeddings are stored in ChromaDB, a vector database designed for efficient similarity search.
4. **Query Input**: The user inputs a query into the chatbot.
5. **Similarity Search**: The query is embedded as a vector and compared with the document embeddings in ChromaDB using a retriever.
6. **Answer Generation**: The most relevant document chunks are retrieved, and the large language model generates a response based on this context.

## Installation
To run this project, you will need Python 3.x installed along with the following libraries:

- Gemini API
- Ngrok API (If using Colab)
- langchain
- langchain-google-genai
- langchain-community
- Streamlit
- ChromaDB
- PyMuPDF
- PyNgrok

You can install the required packages using pip:
```bash
pip install langchain langchain-google-genai langchain-community streamlit chromadb PyMuPDF pyngrok
