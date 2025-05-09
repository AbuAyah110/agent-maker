import os
import json
import yaml  # Import PyYAML
from typing import List, Dict, Any
from pydantic import BaseModel, Field  # Import Pydantic

# Determine the absolute path to the 'prompts' directory
# This assumes manager.py is in agent_maker/prompt_library/
# and the prompts are in agent_maker/prompt_library/prompts/
_PROMPTS_DIR_NAME = "prompts"
PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), _PROMPTS_DIR_NAME
)

# --- Pydantic Models ---
class PromptMetadata(BaseModel):
    version: str = "1.0.0"
    author: str | None = None
    tags: List[str] = []
    # Add any other metadata fields you might want

class PromptData(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        description="Unique name for the prompt (used as filename)."
    )
    description: str | None = None
    template: str = Field(..., min_length=1)
    input_variables: List[str] = []
    metadata: PromptMetadata = Field(default_factory=PromptMetadata)

# --- Exceptions ---
class PromptNotFoundError(Exception):
    pass

class PromptSaveError(Exception):
    pass

# --- Functions ---
def list_prompts() -> List[str]:
    """
    Scans the PROMPTS_DIR for .json, .yaml, and .yml files 
    and returns a list of prompt names (filenames without extensions).
    """
    if not os.path.exists(PROMPTS_DIR):
        os.makedirs(PROMPTS_DIR)  # Create prompts dir if it doesn't exist
        return []

    prompt_names = []
    for f_name in os.listdir(PROMPTS_DIR):
        if f_name.endswith(".json"):
            prompt_names.append(f_name[:-5])  # Remove .json
        elif f_name.endswith(".yaml"):
            prompt_names.append(f_name[:-5])  # Remove .yaml
        elif f_name.endswith(".yml"):
            prompt_names.append(f_name[:-4])  # Remove .yml
            
    # Remove duplicates if somehow a prompt exists as both json and yaml
    return sorted(list(set(prompt_names)))


def load_prompt(prompt_name: str) -> Dict[str, Any]:
    """
    Loads a specific prompt by its name, checking for .json, .yaml, or .yml files.

    Args:
        prompt_name: The name of the prompt to load.

    Returns:
        A dictionary containing the prompt data.

    Raises:
        PromptNotFoundError: If no matching prompt file is found.
        ValueError: If the prompt file is not valid JSON or YAML.
        IOError: If there's an error reading the file.
    """
    base_path = os.path.join(PROMPTS_DIR, prompt_name)
    possible_files = {
        "json": f"{base_path}.json",
        "yaml": f"{base_path}.yaml",
        "yml": f"{base_path}.yml",
    }

    found_file_path = None
    file_type = None

    for f_type, f_path in possible_files.items():
        if os.path.exists(f_path):
            found_file_path = f_path
            file_type = f_type
            break

    if not found_file_path:
        raise PromptNotFoundError(
            f"Prompt '{prompt_name}' not found with .json/.yaml/.yml in "
            f"{PROMPTS_DIR}"
        )

    try:
        with open(found_file_path, 'r') as f:
            if file_type == 'json':
                prompt_data = json.load(f)
            elif file_type in ['yaml', 'yml']:
                prompt_data = yaml.safe_load(f)
            else:
                # Should not happen, but good for safety
                msg = f"Unsupported file type for {found_file_path}"
                raise ValueError(msg)
        return prompt_data
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from '{prompt_name}': {e}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error decoding YAML from '{prompt_name}': {e}")
    except Exception as e:
        msg = f"Error reading prompt file '{prompt_name}' ({found_file_path})"
        raise IOError(f"{msg}: {e}")

def save_prompt(prompt_data: PromptData) -> Dict[str, Any]:
    """
    Saves prompt data to a JSON file in the PROMPTS_DIR.
    Uses the 'name' field from prompt_data for the filename.
    Overwrites existing files with the same name.

    Args:
        prompt_data: A PromptData object containing the prompt details.

    Returns:
        The saved prompt data as a dictionary.

    Raises:
        PromptSaveError: If there is an error saving the file.
    """
    # Basic sanitization for filename might be needed (e.g., no slashes)
    # For now, just use the name directly.
    filename = f"{prompt_data.name}.json"
    file_path = os.path.join(PROMPTS_DIR, filename)

    # Ensure the prompts directory exists
    if not os.path.exists(PROMPTS_DIR):
        try:
            os.makedirs(PROMPTS_DIR)
        except OSError as e:
            msg = f"Could not create prompts directory {PROMPTS_DIR}"
            raise PromptSaveError(f"{msg}: {e}")

    try:
        # Convert Pydantic model to dict for saving
        data_to_save = prompt_data.model_dump()
        
        with open(file_path, 'w') as f:
            json.dump(data_to_save, f, indent=2)
            
        return data_to_save # Return the data that was saved
    except IOError as e:
        raise PromptSaveError(f"Could not write prompt file {file_path}: {e}")
    except Exception as e:
        # Catch other potential errors during model dumping or file writing
        msg = f"An unexpected error occurred while saving prompt '{prompt_data.name}'"
        raise PromptSaveError(f"{msg}: {e}")


if __name__ == '__main__':
    # Ensure PROMPTS_DIR exists for testing, if not already handled by list_prompts
    if not os.path.exists(PROMPTS_DIR):
        os.makedirs(PROMPTS_DIR)

    print(f"Looking for prompts in: {PROMPTS_DIR}")

    # Test listing prompts
    print("\nAvailable prompts (JSON & YAML):")
    available = list_prompts()
    if available:
        for p_name in available:
            print(f"- {p_name}")
    else:
        print("No prompts found.")

    # Test loading prompts
    if available:
        for p_name in available:
            print(f"\nLoading prompt: {p_name}")
            try:
                prompt_content = load_prompt(p_name)
                print(f"Successfully loaded '{p_name}':")
                print(json.dumps(prompt_content, indent=2))
            except (PromptNotFoundError, ValueError, IOError) as e:
                print(f"Error loading prompt '{p_name}': {e}")

    # Test saving a new prompt
    print("\nAttempting to save a new prompt ('test_save_prompt'):")
    new_prompt_data = PromptData(
        name="test_save_prompt",
        description="A prompt saved via the manager test.",
        template="This is a test template for {variable}.",
        input_variables=["variable"],
        metadata=PromptMetadata(tags=["test", "save"])
    )
    try:
        saved_data = save_prompt(new_prompt_data)
        print("Successfully saved prompt:")
        print(json.dumps(saved_data, indent=2))
        # Verify it exists by listing again
        print("\nAvailable prompts after save:")
        available_after_save = list_prompts()
        if available_after_save:
             for p_name in available_after_save:
                 print(f"- {p_name}")
        else:
             print("No prompts found after save.")
        # Clean up the test file
        test_file_path = os.path.join(PROMPTS_DIR, "test_save_prompt.json")
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
            print(f"\nCleaned up test file: {test_file_path}")
    except PromptSaveError as e:
        print(f"Error saving prompt: {e}")

    # Test loading a non-existent prompt
    non_existent = "non_existent_prompt"
    print(f"\nAttempting to load a non-existent prompt ('{non_existent}'):")
    try:
        load_prompt(non_existent)
    except PromptNotFoundError as e:
        print(f"Correctly caught error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}") 