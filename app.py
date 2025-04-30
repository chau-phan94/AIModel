import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from models.ollama_manager import OllamaManager
from services.embedding_service import EmbeddingService
from services.chat_service import ChatService
import os
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
    return OllamaManager.get_local_models()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "deepseek-coder"
if "local_models" not in st.session_state:
    st.session_state.local_models = get_local_models()

# Initialize the LLM
def get_llm(model_name):
    from langchain_community.llms import Ollama
    return Ollama(model=model_name)

# Initialize the embeddings
@st.cache_resource
def get_embeddings():
    return EmbeddingService.get_embeddings()

# Initialize the vector store
@st.cache_resource
def get_vector_store():
    embeddings = get_embeddings()
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Create the RAG chain
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
    from models.ollama_manager import OllamaManager
    local_models = OllamaManager.get_local_models()
    
    if local_models:
        selected_model = st.selectbox(
            "Choose a local model",
            options=local_models,
            key="select_local_model"
        )
        st.session_state.selected_model = selected_model
    else:
        st.warning("No local models found. Please pull a model using Ollama CLI or the button below.")
    
    # Tag a model section
    st.subheader("Tag a Model")
    from models.ollama_manager import OllamaManager
    tag_models = OllamaManager.get_local_models()
    
    if tag_models:
        source_model = st.selectbox(
            "Select source model to tag",
            options=tag_models,
            key="tag_source_model"
        )
        new_tag = st.text_input("Enter new tag (e.g., llama2:custom)", key="new_tag_input")
        if st.button("Tag Model"):
            result = OllamaManager.tag_model(source_model, new_tag)
            if result == 0:
                st.success(f"Model '{source_model}' tagged as '{new_tag}'")
                # Refresh the model list
                tag_models = OllamaManager.get_local_models()
                st.rerun()
            else:
                st.error(f"Failed to tag model '{source_model}'.")
    else:
        st.warning("No local models available to tag.")
    
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
                # Ask user for destination directory to save the model
                destination_path = st.text_input("Enter destination folder to save the model (optional)", key="save_model_dest")
                if destination_path:
                    from models.ollama_manager import OllamaManager
                    save_result = OllamaManager.save_model(model_to_pull, destination_path)
                    if save_result:
                        st.success(f"Model '{model_to_pull}' saved to {destination_path}")
                    else:
                        st.error(f"Failed to save model '{model_to_pull}' to {destination_path}. Please check the path and permissions.")
                get_local_models.clear()  # Clear the cache so new models are detected
                st.session_state.local_models = get_local_models()
                st.rerun()
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
    # Robustly decode the uploaded file to avoid UnicodeDecodeError
    text = uploaded_file.read().decode("utf-8", errors="replace")
    
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

# Website reader UI
st.subheader("Read Website Content")
with st.expander("Extract text from a website", expanded=False):
    url = st.text_input("Enter website URL", key="website_url_input")
    if st.button("Extract Website Text", key="extract_website_text_btn"):
        if url:
            with st.spinner(f"Extracting content from {url}..."):
                try:
                    from services.web_reader_service import fetch_website_text
                    website_text = fetch_website_text(url)
                    preview = website_text[:1000] + ("..." if len(website_text) > 1000 else "")
                    st.success("Website content extracted!")
                    st.text_area("Website Text Preview", preview, height=300)
                    if 'website_text_full' not in st.session_state:
                        st.session_state['website_text_full'] = website_text
                    else:
                        st.session_state['website_text_full'] = website_text
                except Exception as e:
                    st.error(f"Failed to extract website content: {e}")
        else:
            st.warning("Please enter a website URL.")

    # Offer to save extracted text to knowledge base
    if st.session_state.get('website_text_full'):
        if st.button("Save to Knowledge Base", key="save_website_text_btn"):
            with st.spinner("Processing and saving website content..."):
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.split_text(st.session_state['website_text_full'])
                embeddings = get_embeddings()
                vector_store = get_vector_store()
                vector_store.add_texts(chunks)
                st.success("Website content processed and added to knowledge base!")

# Chat interface
st.subheader("Chat with the RAG Agent")

# Show the currently selected model
selected_model = st.session_state.get("selected_model", None)
if selected_model:
    # Make model name more readable (capitalize and replace dashes/underscores with spaces)
    readable_model = selected_model.replace('-', ' ').replace('_', ' ').title()
    st.info(f"**Current Model:** {readable_model}")
else:
    st.warning("No model selected. Please select a model from the sidebar.")

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