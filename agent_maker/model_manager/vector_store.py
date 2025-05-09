from typing import List, Dict, Any, Optional
from langchain.vectorstores import FAISS, Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import os
import shutil


class VectorStoreManager:
    """
    Modular vector store manager supporting FAISS and Chroma.
    """
    def __init__(
        self,
        store_type: str = "faiss",
        embedding_model: Optional[Any] = None,
        persist_directory: Optional[str] = None,
        **kwargs
    ):
        self.store_type = store_type.lower()
        self.persist_directory = persist_directory or "./vector_store_data"
        self.embedding_model = embedding_model or NVIDIAEmbeddings()
        self.vectorstore = None
        self._init_store(**kwargs)

    def _init_store(self, **kwargs):
        if self.store_type == "faiss":
            if os.path.exists(self.persist_directory):
                try:
                    print(f"[VectorStoreManager] Attempting to load FAISS index from: {self.persist_directory}")
                    self.vectorstore = FAISS.load_local(
                        self.persist_directory,
                        self.embedding_model,
                        allow_dangerous_deserialization=True
                    )
                    print(f"[VectorStoreManager] FAISS index loaded successfully from: {self.persist_directory}")
                except FileNotFoundError:
                    print(f"[VectorStoreManager] FAISS index not found at: {self.persist_directory}. Will initialize on add_chunks.")
                    self.vectorstore = None
                except Exception as e:
                    print(f"[VectorStoreManager] Error loading FAISS index from {self.persist_directory}: {e}")
                    self.vectorstore = None # Ensure vectorstore is None if loading failed
            else:
                print(f"[VectorStoreManager] FAISS persist directory not found: {self.persist_directory}. Will initialize on add_chunks.")
                self.vectorstore = None
        elif self.store_type == "chroma":
            os.makedirs(
                self.persist_directory, exist_ok=True
            )
            self.vectorstore = Chroma(
                collection_name="rag_chunks",
                embedding_function=self.embedding_model,
                persist_directory=self.persist_directory,
                **kwargs
            )
        else:
            raise ValueError(
                f"Unsupported vector store type: {self.store_type}"
            )

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk.get("metadata", {}) for chunk in chunks]

        if self.store_type == "faiss":
            if not texts:
                # FAISS cannot be initialized or added to with empty texts
                if self.vectorstore is None:
                    raise ValueError(
                        "No texts provided to initialize FAISS store, and no existing store loaded."
                    )
                else:
                    # If store exists but no new texts, it's a no-op for adding, but still save.
                    print("[VectorStoreManager] No new texts to add to existing FAISS store, but will re-save.")
            
            if self.vectorstore is None and texts: # Only initialize if no store and new texts
                print(f"[VectorStoreManager] Initializing new FAISS store with {len(texts)} texts.")
                self.vectorstore = FAISS.from_texts(
                    texts, self.embedding_model, metadatas=metadatas
                )
            elif texts: # Store exists, add new texts
                print(f"[VectorStoreManager] Adding {len(texts)} texts to existing FAISS store.")
                self.vectorstore.add_texts(texts, metadatas=metadatas)

            if self.vectorstore: # If store exists (either loaded or newly created with texts)
                try:
                    # Ensure clean directory for saving FAISS
                    if os.path.exists(self.persist_directory):
                        print(f"[VectorStoreManager] Removing existing FAISS directory for clean save: {self.persist_directory}")
                        shutil.rmtree(self.persist_directory)
                    os.makedirs(self.persist_directory, exist_ok=True)
                    print(f"[VectorStoreManager] Saving FAISS index to: {self.persist_directory}")
                    self.vectorstore.save_local(self.persist_directory)
                    print(f"[VectorStoreManager] FAISS index saved successfully to: {self.persist_directory}")
                except Exception as e:
                    print(f"[VectorStoreManager] Error saving FAISS index to {self.persist_directory}: {e}")
            else:
                # This case should ideally be caught by the "No texts provided" error earlier if creating new
                # or imply an issue with loading an existing store that wasn't re-initialized.
                print("[VectorStoreManager] FAISS store is None after add_chunks logic, cannot save.")

        elif self.store_type == "chroma":
            if not texts and self.vectorstore.get()['ids'] == []: # Chroma specific check for empty
                 raise ValueError("No texts provided to initialize or add to Chroma store.")
            if texts:
                self.vectorstore.add_texts(texts, metadatas=metadatas)
            self.vectorstore.persist()

    def search(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ):
        if self.vectorstore is None:
            raise ValueError("Vector store is not initialized. Index chunks first.")
        # LangChain's vectorstores support metadata filtering for Chroma
        if self.store_type == "chroma" and filters:
            results = self.vectorstore.similarity_search_with_score(
                query, k=top_k, filter=filters
            )
        else:
            results = self.vectorstore.similarity_search_with_score(
                query, k=top_k
            )
        # Each result: (Document, score)
        # Convert score to float for JSON serialization
        return [
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score) if score is not None else None
            }
            for doc, score in results
        ]

    def hybrid_search(
        self,
        query: str,
        keyword: Optional[str] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ):
        if self.vectorstore is None:
            raise ValueError("Vector store is not initialized. Index chunks first.")
        """
        Hybrid search: combines vector similarity and keyword filtering.
        For Chroma, uses both vector and keyword (where_document/where) filters.
        For FAISS, filters vector search results by keyword in chunk text.
        """
        if self.store_type == "chroma":
            chroma_kwargs = {}
            if filters:
                chroma_kwargs["filter"] = filters
            if keyword:
                chroma_kwargs["where_document"] = {"$contains": keyword}
            results = self.vectorstore.similarity_search_with_score(
                query, k=top_k, **chroma_kwargs
            )
        else:  # FAISS
            results = self.vectorstore.similarity_search_with_score(
                query, k=top_k
            )
            if keyword:
                results = [
                    (doc, score)
                    for doc, score in results
                    if keyword.lower() in doc.page_content.lower()
                ]
        # Convert score to float for JSON serialization
        return [
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score) if score is not None else None
            }
            for doc, score in results
        ]

    def list_chunks(self):
        # For Chroma, can use get() to list all docs; for FAISS, not directly supported
        if self.store_type == "chroma":
            # Chroma exposes get() for all docs
            all_docs = self.vectorstore.get()
            return [
                {"text": text, "metadata": meta}
                for text, meta in zip(
                    all_docs["documents"],
                    all_docs["metadatas"]
                )
            ]
        else:
            # For FAISS, we can only list what was added in this session
            # (FAISS does not persist or expose all docs by default)
            return []

    def delete_chunks(self, filter: Optional[Dict[str, Any]] = None, ids: Optional[list] = None):
        """
        Delete chunks by metadata filter or document IDs. Only supported for Chroma.
        """
        if self.store_type != "chroma":
            raise NotImplementedError("Delete is only supported for Chroma.")
        if ids:
            self.vectorstore.delete(ids=ids)
        elif filter:
            self.vectorstore.delete(where=filter)
        self.vectorstore.persist()

    def update_chunk(self, doc_id: str, new_text: str = None, new_metadata: dict = None):
        """
        Update a chunk by document ID. Only supported for Chroma.
        """
        if self.store_type != "chroma":
            raise NotImplementedError("Update is only supported for Chroma.")
        # Chroma does not support in-place update; delete and re-add
        docs = self.vectorstore.get(ids=[doc_id])
        if not docs["documents"]:
            raise ValueError(f"Document with id {doc_id} not found.")
        old_metadata = docs["metadatas"][0]
        new_meta = old_metadata.copy()
        if new_metadata:
            new_meta.update(new_metadata)
        self.vectorstore.delete(ids=[doc_id])
        self.vectorstore.add_texts([new_text or docs["documents"][0]], metadatas=[new_meta], ids=[doc_id])
        self.vectorstore.persist()

# Example usage:
# vsm = VectorStoreManager(store_type="chroma")
# vsm.add_chunks([{"text": "hello world", "metadata": {"doc_id": 1}}])
# results = vsm.hybrid_search("hello", keyword="world", top_k=3)
# vsm.delete_chunks(filter={"doc_id": 1})
# vsm.update_chunk(doc_id="abc123", new_text="updated text") 