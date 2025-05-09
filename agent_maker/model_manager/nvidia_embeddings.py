from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

class NvidiaEmbeddingService:
    def __init__(self, model="NV-Embed-QA", base_url=None, api_key=None, truncate="NONE"):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.truncate = truncate
        self.embedder = NVIDIAEmbeddings(
            model=model,
            base_url=base_url,
            api_key=api_key,
            truncate=truncate
        )

    def embed_query(self, text):
        return self.embedder.embed_query(text)

    def embed_documents(self, texts):
        return self.embedder.embed_documents(texts)

    async def aembed_query(self, text):
        return await self.embedder.aembed_query(text)

    async def aembed_documents(self, texts):
        return await self.embedder.aembed_documents(texts) 