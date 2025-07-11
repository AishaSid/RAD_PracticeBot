# Gemini PDF Summarizer

RAG -> Retrieval Augmented Generation

It loads data from uploaded PDF and stores it in a vectore store for fast and relevant (data related to user query) data retrieval, then it uses an LLM (Gemini) to generate language to answer the user queries.  

## Overview

- Upload PDF documents
- Convert PDF content into vector embeddings using Google Gemini
- Store embeddings in a FAISS vector database
- Ask questions about the PDF content and receive AI-powered answers
- Retrieve relevant information from processed documents
- Interactive question-answering experience

## Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install streamlit
pip install google-generativeai
pip install python-dotenv
pip install langchain
pip install langchain-community
pip install langchain-google-genai
pip install faiss-cpu
pip install pypdf
```

### 3. Environment Setup

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_ai_studio_api_key_here
```

To get your API key:
1. Visit [Google AI Studio](https://makersuite.google.com/)
2. Create an account or sign in
3. Generate a new API key
4. Copy the key to your `.env` file

### Running the Application

```bash
streamlit run app.py
```

### Components Used

- **Streamlit**: Web application framework
- **Google Gemini**: Large language model for embeddings and chat
- **LangChain**: Framework for building LLM applications
- **FAISS**: Vector similarity search library
- **PyPDF**: PDF text extraction

### How It Works

1. **PDF Upload**: User uploads a PDF file through the Streamlit interface
2. **Text Extraction**: PyPDF extracts text content from the PDF
3. **Text Chunking**: LangChain splits text into manageable chunks
4. **Vector Embeddings**: Google Gemini generates embeddings for each chunk
5. **Vector Storage**: FAISS stores embeddings for efficient retrieval
6. **Question Processing**: User questions are converted to embeddings
7. **Similarity Search**: FAISS finds relevant text chunks
8. **Answer Generation**: Gemini generates answers based on retrieved context
