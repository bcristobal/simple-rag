from abc import ABC, abstractmethod
from typing import List, AsyncGenerator, Any
from srag.core import BaseLLM, BaseEmbedder, BaseVectorStore, Chunk

class BaseRAGStrategy(ABC):
    """
    Clase base abstracta para estrategias de RAG.
    """
    
    def __init__(self, llm: BaseLLM, embedder: BaseEmbedder, vector_store: BaseVectorStore):
        self.llm = llm
        self.embedder = embedder
        self.vector_store = vector_store

    @abstractmethod
    async def retrieve(self, query: str, k: int = 4, **kwargs) -> List[Chunk]:
        """Lógica para encontrar documentos relevantes."""
        pass

    @abstractmethod
    async def stream(self, query: str, k: int = 4, **kwargs) -> AsyncGenerator[str, None]:
        """Genera una respuesta en streaming."""
        pass

    # Método helper común para todas las estrategias
    def _build_context(self, chunks: List[Chunk]) -> str:
        """Convierte una lista de chunks en un string de contexto único."""
        return "\n\n".join([f"--- Fragmento {i+1} ---\n{c.content}" for i, c in enumerate(chunks)])