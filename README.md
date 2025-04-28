# DeepSeek RAG Agent

A local RAG (Retrieval-Augmented Generation) agent powered by DeepSeek AI, Ollama, and NOMIC embeddings.

## Features

- 🤖 Powered by DeepSeek R1 (7B) model via Ollama
- 🔍 NOMIC embeddings for semantic search
- 💾 ChromaDB for vector storage
- 🎯 User-friendly Streamlit interface
- 📚 Document upload and processing
- 💬 Interactive chat interface

## Prerequisites

1. Install [Ollama](https://ollama.com)
2. Python 3.8 or higher

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Pull the required models:
```bash
ollama pull deepseek-coder:7b
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Upload documents to build your knowledge base:
   - Click the "Upload a document" button
   - Select a text file (.txt), PDF (.pdf), or Markdown (.md) file
   - Wait for the document to be processed

4. Start chatting with the RAG agent:
   - Type your questions in the chat input
   - The agent will use the uploaded documents to provide relevant answers

## How it Works

1. **Document Processing**:
   - Documents are split into chunks
   - Chunks are converted into embeddings using NOMIC
   - Embeddings are stored in ChromaDB

2. **Question Answering**:
   - User questions are converted into embeddings
   - Similar chunks are retrieved from the vector store
   - The DeepSeek model generates answers based on the retrieved context

## Notes

- The application runs completely locally
- No API keys or external services are required
- All data is stored in the `./chroma_db` directory

## License

MIT License 