from typing import List, Dict, Any, Optional
import os
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
    CSVLoader,
    JSONLoader,
    # Add more as needed
)

# Map loader type to LangChain loader class
LOADER_MAP = {
    "text": TextLoader,
    "pdf": PyPDFLoader,  # or UnstructuredPDFLoader
    "markdown": UnstructuredMarkdownLoader,
    "html": WebBaseLoader,
    "csv": CSVLoader,
    "json": JSONLoader,
    # Add more mappings here
}

# Map file extension to loader type
EXTENSION_MAP = {
    ".txt": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".pdf": "pdf",
    ".html": "html",
    ".htm": "html",
    ".csv": "csv",
    ".json": "json",
    # Add more as needed
}


def detect_loader_type(file_path: str) -> Optional[str]:
    _, ext = os.path.splitext(file_path.lower())
    return EXTENSION_MAP.get(ext)


def load_document(
    file_path: str,
    loader_type: Optional[str] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Loads a document using the specified or auto-detected loader type.
    Returns a list of dicts: [{"text": ..., "metadata": ...}, ...]
    """
    if loader_type is None:
        loader_type = detect_loader_type(file_path)
        if loader_type is None:
            raise ValueError(
                f"Could not auto-detect loader type for: {file_path}"
            )
    loader_cls = LOADER_MAP.get(loader_type)
    if not loader_cls:
        raise ValueError(f"Unsupported loader type: {loader_type}")
    # Some loaders (e.g., WebBaseLoader) expect a URL, not a file path
    if loader_type == "html" and (
        file_path.startswith("http://") or file_path.startswith("https://")
    ):
        loader = loader_cls(file_path, **kwargs)
    else:
        loader = loader_cls(file_path, **kwargs)
    docs = loader.load()
    # Each doc is a Document object with .page_content and .metadata
    return [
        {"text": doc.page_content, "metadata": doc.metadata} for doc in docs
    ]

# Example usage:
# docs = load_document("myfile.pdf")
# docs = load_document("myfile.txt")
# docs = load_document("myfile.md") 