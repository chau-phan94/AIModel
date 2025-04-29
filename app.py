import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import os
import subprocess
import json

# Popular models for easy selection
POPULAR_MODELS = [
    "deepseek-coder",
    "llama2",
    "mistral",
    "codellama",
    "neural-chat",
    "starling-lm"
]

# Set page config
st.set_page_config(
    page_title="DeepSeek RAG Agent",
    page_icon="🤖",
    layout="wide"
)

# Helper to get local models from Ollama
@st.cache_data(show_spinner=False)
def get_local_models():
    try:
        result = subprocess.run(["ollama", "list", "--json"], capture_output=True, text=True)
        models = []
        for line in result.stdout.strip().splitlines():
            try:
                model_info = json.loads(line)
                models.append(model_info["name"])
            except Exception:
                continue
        return models
    except Exception:
        return []

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "deepseek-coder"
if "local_models" not in st.session_state:
    st.session_state.local_models = get_local_models()

# Initialize the LLM
@st.cache_resource
def get_llm(model_name):
    return Ollama(model=model_name)

# Initialize the embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize the vector store
@st.cache_resource
def get_vector_store():
    embeddings = get_embeddings()
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Create the RAG chain
@st.cache_resource
def get_rag_chain(model_name):
    llm = get_llm(model_name)
    vector_store = get_vector_store()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3})
    )

# Main UI
st.title("🤖 DeepSeek RAG Agent")
st.markdown("""
This is a local RAG (Retrieval-Augmented Generation) agent powered by:
- Ollama models
- HuggingFace embeddings
- ChromaDB for vector storage
""")

# Sidebar with settings
with st.sidebar:
    st.header("Settings")
    
    # Refresh local models
    if st.button("Refresh Local Models"):
        st.session_state.local_models = get_local_models()
        st.success("Model list refreshed!")
    
    # Model selection from local models
    st.subheader("Select Local Model")
    if st.session_state.local_models:
        selected_model = st.selectbox(
            "Choose a local model",
            options=st.session_state.local_models,
            index=st.session_state.local_models.index(st.session_state.selected_model) if st.session_state.selected_model in st.session_state.local_models else 0,
            help="Select a model you have already pulled with Ollama",
            key="select_local_model"
        )
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            st.session_state.chat_history = []
    else:
        st.warning("No local models found. Please pull a model using Ollama CLI or the button below.")
    
    # Model tagging (renaming)
    st.subheader("Tag (Rename) a Model")
    with st.form("tag_model_form"):
        source_model = st.selectbox("Source model", options=st.session_state.local_models, key="tag_source_model")
        new_tag = st.text_input("New tag name (no spaces)")
        tag_submitted = st.form_submit_button("Tag Model")
        if tag_submitted and new_tag:
            cmd = f"ollama copy {source_model} {new_tag}"
            result = os.system(cmd)
            if result == 0:
                st.success(f"Tagged {source_model} as {new_tag}")
                st.session_state.local_models = get_local_models()
            else:
                st.error("Failed to tag model. Make sure the name is valid and not already used.")
    
    # Pull model with dropdown
    st.subheader("Pull a Model")
    pull_model_choice = st.selectbox(
        "Choose a model to pull",
        options=POPULAR_MODELS,
        key="pull_model_choice"
    )
    custom_model_name = st.text_input("Or enter a custom model name (optional)")
    if st.button("Pull Model"):
        model_to_pull = custom_model_name.strip() if custom_model_name.strip() else pull_model_choice
        with st.spinner(f"Pulling {model_to_pull}..."):
            result = os.system(f"ollama pull {model_to_pull}")
            if result == 0:
                st.success(f"Successfully pulled {model_to_pull}")
                st.session_state.local_models = get_local_models()
            else:
                st.error(f"Error pulling model: {model_to_pull}")
    
    # Clear chat history button
    st.subheader("Chat Management")
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # About section
    st.header("About")
    st.markdown("""
    This RAG agent uses:
    - Ollama models for reasoning
    - HuggingFace embeddings for semantic search
    - ChromaDB for vector storage
    - Streamlit for the interface
    
    Upload documents to build your knowledge base and ask questions!
    """)

# File uploader for documents
uploaded_file = st.file_uploader("Upload a document to add to the knowledge base", type=['txt', 'pdf', 'md'])

if uploaded_file is not None:
    # Process the uploaded file
    text = uploaded_file.read().decode()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    
    # Create embeddings and store in vector store
    embeddings = get_embeddings()
    vector_store = get_vector_store()
    vector_store.add_texts(chunks)
    
    st.success("Document processed and added to knowledge base!")

# Chat interface
st.subheader("Chat with the RAG Agent")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get RAG response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            rag_chain = get_rag_chain(st.session_state.selected_model)
            response = rag_chain.invoke({"query": prompt})
            st.write(response["result"])
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response["result"]}) 