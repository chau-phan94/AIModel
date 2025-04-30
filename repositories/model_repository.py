from models.ollama_manager import OllamaManager
import os

class ModelRepository:
    @staticmethod
    def get_local_models():
        return OllamaManager.get_local_models()

    @staticmethod
    def tag_model(source_model, new_tag):
        return OllamaManager.tag_model(source_model, new_tag)

    @staticmethod
    def pull_model(model_name):
        return os.system(f"ollama pull {model_name}")

    @staticmethod
    def save_model(model_name, destination_path):
        return OllamaManager.save_model(model_name, destination_path)
