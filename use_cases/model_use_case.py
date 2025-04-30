from repositories.model_repository import ModelRepository

class ModelUseCase:
    @staticmethod
    def list_local_models():
        return ModelRepository.get_local_models()

    @staticmethod
    def tag_model(source_model, new_tag):
        return ModelRepository.tag_model(source_model, new_tag)

    @staticmethod
    def pull_model(model_name):
        return ModelRepository.pull_model(model_name)

    @staticmethod
    def save_model(model_name, destination_path):
        return ModelRepository.save_model(model_name, destination_path)
