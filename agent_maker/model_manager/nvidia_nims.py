# Placeholder for NVIDIA NIMs model management

import os
import getpass
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from agent_maker.config import settings  # Import the settings


def get_available_nvidia_models():
    """
    Fetches the list of available NVIDIA models using LangChain.
    Tries to load NVIDIA_API_KEY from config, then env, then prompts.
    """
    api_key = settings.NVIDIA_API_KEY  # Try from config first

    if not api_key:
        # Try from environment directly as a fallback
        api_key = os.getenv("NVIDIA_API_KEY")

    if not api_key:
        print("NVIDIA_API_KEY not found in config or environment variables.")
        if not os.isatty(0):  # Check if non-interactive
            # For non-interactive use, API key must be pre-configured
            raise ValueError(
                "NVIDIA_API_KEY must be set for non-interactive use."
            )
        try:
            prompted_key = getpass.getpass(
                "Enter your NVIDIA API key (starts with 'nvapi-'): "
            )
            if not prompted_key.startswith("nvapi-"):
                print("Invalid API key format. Must start with 'nvapi-'.")
                return []
            os.environ["NVIDIA_API_KEY"] = prompted_key  # Set for this session
            api_key = prompted_key
        except Exception as e:
            print(f"Could not get API key via prompt: {e}")
            return []
    else:
        # If key was found, ensure it's in the env for ChatNVIDIA
        if not os.getenv("NVIDIA_API_KEY"):
            os.environ["NVIDIA_API_KEY"] = api_key

    if not api_key:  # Final check if API key is still missing
        print("NVIDIA API Key is required to fetch models.")
        return []

    try:
        available_models = ChatNVIDIA.get_available_models()
        model_ids = [model.id for model in available_models]
        return model_ids
    except Exception as e:
        print(f"Error fetching NVIDIA models: {e}")
        print(
            "Please ensure your NVIDIA_API_KEY is correct, loaded, and "
            "you have the necessary permissions."
        )
        return []


if __name__ == '__main__':
    print("Attempting to fetch available NVIDIA models...")
    models = get_available_nvidia_models()
    if models:
        print("Available NVIDIA Models:")
        for model_id in models:
            print(f"- {model_id}")
    else:
        print("No models found or an error occurred.") 