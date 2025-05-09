from enum import Enum
from typing import List, Dict, Any

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
    # SemanticChunker,  # Experimental, may require extra install
    # CodeTextSplitter, # If you want code support
)


class ChunkingStrategy(str, Enum):
    RECURSIVE = "recursive"
    CHARACTER = "character"
    TOKEN = "token"
    MARKDOWN = "markdown"
    HTML = "html"
    # SEMANTIC = "semantic"
    # CODE = "code"


class Chunker:
    """
    Modular chunker supporting multiple strategies via LangChain splitters.
    """
    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        **kwargs
    ):
        self.strategy = strategy
        self.splitter = self._get_splitter(**kwargs)

    def _get_splitter(self, **kwargs):
        if self.strategy == ChunkingStrategy.RECURSIVE:
            return RecursiveCharacterTextSplitter(**kwargs)
        elif self.strategy == ChunkingStrategy.CHARACTER:
            return CharacterTextSplitter(**kwargs)
        elif self.strategy == ChunkingStrategy.TOKEN:
            return TokenTextSplitter(**kwargs)
        elif self.strategy == ChunkingStrategy.MARKDOWN:
            return MarkdownHeaderTextSplitter(**kwargs)
        elif self.strategy == ChunkingStrategy.HTML:
            return HTMLHeaderTextSplitter(**kwargs)
        # elif self.strategy == ChunkingStrategy.SEMANTIC:
        #     return SemanticChunker(**kwargs)
        # elif self.strategy == ChunkingStrategy.CODE:
        #     return CodeTextSplitter(**kwargs)
        else:
            raise ValueError(
                f"Unsupported chunking strategy: {self.strategy}"
            )

    def chunk_text(
        self, text: str, metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Splits text into chunks. Returns a list of dicts:
        [{"text": ..., "metadata": ...}, ...]
        """
        if metadata is None:
            metadata = {}
        return [
            {"text": chunk, "metadata": metadata.copy()}
            for chunk in self.splitter.split_text(text)
        ]

# Example usage:
# chunker = Chunker(
#     strategy=ChunkingStrategy.RECURSIVE, chunk_size=512, chunk_overlap=50
# )
# chunks = chunker.chunk_text(
#     "Your long document text here...", metadata={"doc_id": "123"}
# ) 