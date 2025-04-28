import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import os

# Set page config
st.set_page_config(
    page_title="DeepSeek RAG Agent",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize the LLM
@st.cache_resource
def get_llm():
    return Ollama(model="deepseek-coder:7b")

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
def get_rag_chain():
    llm = get_llm()
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
- DeepSeek R1 (7B) model via Ollama
- HuggingFace embeddings
- ChromaDB for vector storage
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
            rag_chain = get_rag_chain()
            response = rag_chain.invoke({"query": prompt})
            st.write(response["result"])
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response["result"]})

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This RAG agent uses:
    - DeepSeek R1 (7B) model for reasoning
    - HuggingFace embeddings for semantic search
    - ChromaDB for vector storage
    - Streamlit for the interface
    
    Upload documents to build your knowledge base and ask questions!
    """)
    
    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun() 