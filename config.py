import os

def load_env_file():
    env_vars = {}
    # Search in current directory and parent directories
    paths = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(os.getcwd()), ".env"),
        r"c:\Users\MUHAMMAD AHMAD\Downloads\LLMasW\.env"
    ]
    for path in paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split('=', 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            val = parts[1].strip()
                            # Remove quotes if present
                            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                                val = val[1:-1]
                            env_vars[key] = val
            break
    return env_vars

# Load keys
env = load_env_file()

OPENROUTER_API_1 = env.get("OPENROUTER_API_1") or os.getenv("OPENROUTER_API_1")
OPENROUTER_API_2 = env.get("OPENROUTER_API_2") or os.getenv("OPENROUTER_API_2")
OPENROUTER_API_KEY = env.get("OPENROUTER_API") or os.getenv("OPENROUTER_API")

OPENROUTER_API_KEYS = []
if OPENROUTER_API_1:
    OPENROUTER_API_KEYS.append(OPENROUTER_API_1)
if OPENROUTER_API_2:
    OPENROUTER_API_KEYS.append(OPENROUTER_API_2)
if OPENROUTER_API_KEY and OPENROUTER_API_KEY not in OPENROUTER_API_KEYS:
    OPENROUTER_API_KEYS.append(OPENROUTER_API_KEY)

OPENROUTER_MODEL = env.get("OPENROUTER_MODEL") or os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
CENSUS_API_KEY = env.get("CENSUS_API_KEY") or os.getenv("CENSUS_API_KEY")

LLAMA_CPP_URL = env.get("LLAMA_CPP_URL") or os.getenv("LLAMA_CPP_URL")
LLAMA_CPP_MODEL = env.get("LLAMA_CPP_MODEL") or os.getenv("LLAMA_CPP_MODEL")

if LLAMA_CPP_URL:
    print(f"Llama.cpp Mode: Using local server at {LLAMA_CPP_URL}")
else:
    if not OPENROUTER_API_KEYS:
        print("Warning: No OpenRouter API keys found in environment or .env file.")

if not CENSUS_API_KEY:
    print("Warning: CENSUS_API_KEY not found in environment or .env file.")
