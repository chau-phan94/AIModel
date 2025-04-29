import subprocess
import json
import os

import shutil

class OllamaManager:
    @staticmethod
    def get_local_models():
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            models = []
            for line in result.stdout.strip().splitlines():
                # Skip header and empty lines
                if line.startswith("NAME") or line.strip() == "":
                    continue
                # The first column is the model name (possibly with :tag)
                model_name = line.split()[0]
                models.append(model_name)
            return models
        except Exception as e:
            print(f"Error getting local models: {e}")
            return []

    @staticmethod
    def tag_model(source_model, new_tag):
        cmd = f"ollama copy {source_model} {new_tag}"
        result = os.system(cmd)
        return result == 0

    @staticmethod
    def pull_model(model_name):
        cmd = f"ollama pull {model_name}"
        result = os.system(cmd)
        return result == 0

    @staticmethod
    def save_model(model_name, destination_path):
        """
        Copy the pulled model from Ollama's storage to the user-specified destination folder.
        Works for macOS. Adjust the ollama_model_dir for other OSes if needed.
        """
        import os
        ollama_model_dir = os.path.expanduser('~/Library/Application Support/ollama/models')
        src = os.path.join(ollama_model_dir, model_name)
        dst = os.path.join(destination_path, model_name)
        if os.path.exists(src):
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
            return True
        return False
