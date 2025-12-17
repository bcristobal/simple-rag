from typing import List, AsyncGenerator
from .base import BaseRAGStrategy
from srag.core import Chunk

class SimpleRAG(BaseRAGStrategy):
    """
    Estrategia de RAG Simple:
    1. Convierte Query a Vector.
    2. Busca los K vecinos más cercanos (Similitud del Coseno).
    3. Pasa el contexto al LLM.
    """

    async def retrieve(self, query: str, k: int = 4, **kwargs) -> List[Chunk]:
        print(f"🔍 [SimpleRAG] Buscando: '{query}'")
        query_vector = await self.embedder.embed_query(query)
        return await self.vector_store.search(query_vector, k=k)

    async def stream(self, query: str, k: int = 4, **kwargs) -> AsyncGenerator[str, None]:
        # 1. Retrieve
        relevant_chunks = await self.retrieve(query, k=k)
        
        if not relevant_chunks:
            yield "No encontré información relevante en la base de datos."
            return

        # 2. Augment (Contexto)
        context_text = self._build_context(relevant_chunks)
        
        system_prompt = f"""Eres un asistente experto. Responde a la pregunta basándote ÚNICAMENTE en el siguiente contexto.
        
CONTEXTO:
{context_text}

PREGUNTA: {query}
"""
        # 3. Generate
        async for token in self.llm.stream(system_prompt):
            yield token