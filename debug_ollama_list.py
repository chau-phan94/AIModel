import subprocess

def debug_ollama_list():
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    print("DEBUG RAW OLLAMA OUTPUT:")
    print(result.stdout)
    print("DEBUG SPLIT LINES:")
    for line in result.stdout.strip().splitlines():
        print(f"'{line}'")
    print("DEBUG PARSED MODELS:")
    models = []
    for line in result.stdout.strip().splitlines():
        if line.strip().startswith("NAME") or line.strip() == "":
            continue
        line = line.strip()
        if ':' not in line:
            continue
        model_name = line.split()[0]
        print(f"MODEL: '{model_name}'")
        models.append(model_name)
    print("FINAL MODELS LIST:", models)

if __name__ == "__main__":
    debug_ollama_list()
