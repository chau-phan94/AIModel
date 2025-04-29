# DeepSeek RAG Agent

A local RAG (Retrieval-Augmented Generation) agent powered by DeepSeek Coder, Ollama, and Sentence Transformers embeddings.

## Features

- 🤖 Powered by DeepSeek Coder 7B via Ollama
- 🔍 Sentence Transformers (`all-MiniLM-L6-v2`) for semantic search
- 💾 ChromaDB for fast vector storage
- 🎯 User-friendly Streamlit interface
- 📚 Robust document upload and processing (handles encoding issues)
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
python -m venv venv  # or: python -m venv .venv
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
   - The app robustly handles encoding issues, so problematic files will not crash the app
   - Wait for the document to be processed

4. Start chatting with the RAG agent:
   - Type your questions in the chat input
   - The agent will use the uploaded documents to provide relevant answers

## How it Works

1. **Document Upload:** Upload your knowledge files (txt, pdf, md). The app reads and splits them into chunks, handling encoding issues gracefully.
2. **Embedding:** Each chunk is embedded using Sentence Transformers (`all-MiniLM-L6-v2`).
3. **Vector Storage:** Embeddings are stored in ChromaDB for fast similarity search.
4. **Retrieval:** When you ask a question, relevant chunks are retrieved from the vector store.
5. **LLM Response:** The DeepSeek Coder 7B model (via Ollama) generates an answer using the retrieved context.

## Notes on Large Files

- The `venv/` directory, `.venv/`, and large model files are **not tracked by git** (see `.gitignore`).
- You must pull models locally using Ollama (`ollama pull deepseek-coder:7b`).
- If you encounter errors about large files when pushing to GitHub, see the Troubleshooting section below.

## Troubleshooting

- **Large file push errors:**
    - If you see errors about files over 100MB, it means a large file was committed to git history.
    - Remove it from history using `git filter-repo` (see below) and force-push:
      ```bash
      git filter-repo --path <path-to-large-file> --invert-paths --force
      git remote add origin <your-repo-url>  # if needed
      git push --force origin main
      ```
- **Model not found:**
    - Run `ollama pull deepseek-coder:7b` to download the required model locally.
- **Encoding errors on upload:**
    - The app uses robust decoding (`errors='replace'`) so problematic files will not crash the app. If you see garbled text, check your file’s encoding.

## Requirements

Make sure your `requirements.txt` includes:
- streamlit
- langchain
- chromadb
- sentence-transformers
- (and any other dependencies your code uses)

---

Feel free to open an issue or PR if you need help or want to contribute!


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