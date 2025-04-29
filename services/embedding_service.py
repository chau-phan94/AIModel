from langchain_community.embeddings import HuggingFaceEmbeddings

class EmbeddingService:
    @staticmethod
    def get_embeddings():
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
