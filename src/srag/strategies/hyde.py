from typing import List, AsyncGenerator
from .base import BaseRAGStrategy
from srag.core import Chunk

class HydeRAG(BaseRAGStrategy):
    """
    Estrategia HyDE (Hypothetical Document Embeddings).
    Mejora la recuperación generando una respuesta hipotética y usándola para la búsqueda vectorial.
    """

    async def retrieve(self, query: str, k: int = 4, **kwargs) -> List[Chunk]:
        print(f"👻 [HyDE] Generando documento hipotético para: '{query}'")
        
        # 1. Generar respuesta hipotética
        hyde_prompt = f"""Escribe un pasaje breve que responda a la siguiente pregunta. 
No importa si la información es inventada, solo importa la estructura y el vocabulario técnico.

Pregunta: {query}
Pasaje:"""
        
        hypothetical_doc = await self.llm.generate(hyde_prompt)
        print(f"   -> Hipótesis: {hypothetical_doc[:60]}...")

        # 2. Embed de la hipótesis (NO de la query original)
        query_vector = await self.embedder.embed_query(hypothetical_doc)
        
        # 3. Búsqueda con ese vector
        return await self.vector_store.search(query_vector, k=k)

    async def stream(self, query: str, k: int = 4, **kwargs) -> AsyncGenerator[str, None]:
        # El flujo es igual al SimpleRAG, la magia está en el retrieve()
        relevant_chunks = await self.retrieve(query, k=k)
        
        context_text = self._build_context(relevant_chunks)
        
        prompt = f"""Responde basándote en el contexto real recuperado.

CONTEXTO REAL:
{context_text}

PREGUNTA: {query}
RESPUESTA:"""
        
        async for token in self.llm.stream(prompt):
            yield token