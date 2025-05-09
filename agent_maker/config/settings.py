import os
from dotenv import load_dotenv

# Load environment variables from .env file in the workspace root.
# Adjust the path if your .env file is located elsewhere.
# e.g., for .env in project root (one level above agent_maker directory):
# dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')

# Assuming .env file is in the root of the workspace
# (e.g., /Users/mraza/Documents/src/agent_maker/.env)
# This is two levels up from agent_maker/config/settings.py
_current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(_current_dir, '..', '..'))
dotenv_path = os.path.join(project_root, '.env')

load_dotenv(dotenv_path=dotenv_path)

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if __name__ == '__main__':
    # Test if the key is loaded
    if NVIDIA_API_KEY:
        # Print first 10 chars for security, if key is long enough
        key_len = len(NVIDIA_API_KEY)
        display_key = NVIDIA_API_KEY[:10] + "..." if key_len > 10 else NVIDIA_API_KEY
        print(f"NVIDIA API Key loaded: {display_key}")
    else:
        env_file_path = os.path.join(project_root, ".env")
        print(
            "NVIDIA API Key not found. Make sure you have a .env file "
            f"in the project root (e.g., {env_file_path}) "
            "with NVIDIA_API_KEY set."
        ) 