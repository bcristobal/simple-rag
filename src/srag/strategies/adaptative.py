from typing import List, AsyncGenerator
from .base import BaseRAGStrategy
from srag.core import Chunk

class AdaptiveRAG(BaseRAGStrategy):
    """
    Estrategia Adaptativa:
    1. Analiza la pregunta con el LLM.
    2. Clasifica si necesita recuperación externa (RAG) o no.
    3. Si NO necesita -> Responde directo (Chat mode).
    4. Si SÍ necesita -> Ejecuta flujo de recuperación (Vector Search).
    """

    async def _classify_query(self, query: str) -> bool:
        """Determina si la consulta necesita RAG. Retorna True si necesita contexto."""
        prompt = f"""Eres un clasificador de consultas. Tu trabajo es determinar si una pregunta requiere buscar información externa (documentos, datos, hechos específicos) o si es una pregunta conversacional/general que puedes responder tú solo.

Pregunta: "{query}"

Responde SOLO con una palabra: "BUSCAR" o "RESPONDER"."""
        
        # Usamos generate (no stream) porque necesitamos la decisión completa
        decision = await self.llm.generate(prompt)
        clean_decision = decision.strip().upper()
        
        print(f"🤖 [Adaptive] Decisión: {clean_decision}")
        return "BUSCAR" in clean_decision

    async def retrieve(self, query: str, k: int = 4, **kwargs) -> List[Chunk]:
        # Esta estrategia delega el retrieve. Si se llama directamente, asume búsqueda vectorial.
        query_vector = await self.embedder.embed_query(query)
        return await self.vector_store.search(query_vector, k=k)

    async def stream(self, query: str, k: int = 4, **kwargs) -> AsyncGenerator[str, None]:
        # 1. Clasificación
        needs_retrieval = await self._classify_query(query)

        if not needs_retrieval:
            # FLUJO DIRECTO (Sin RAG)
            print("⚡ [Adaptive] Respondiendo directamente...")
            async for token in self.llm.stream(query):
                yield token
        else:
            # FLUJO RAG (Standard)
            print("📚 [Adaptive] Iniciando recuperación de información...")
            relevant_chunks = await self.retrieve(query, k=k)
            context_text = self._build_context(relevant_chunks)
            
            prompt = f"""Usa el siguiente contexto para responder.
            
CONTEXTO:
{context_text}

PREGUNTA: {query}
RESPUESTA:"""
            
            async for token in self.llm.stream(prompt):
                yield token