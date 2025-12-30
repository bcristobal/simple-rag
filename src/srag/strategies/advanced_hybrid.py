import asyncio
from typing import List, AsyncGenerator, Dict
from sentence_transformers import CrossEncoder
from .base import BaseRAGStrategy
from srag.core import Chunk

class AdvancedHybridRAG(BaseRAGStrategy):
    """
    Estrategia RAG Avanzada (Retrieve & Rerank).
    
    Mejoras sobre el Hybrid original:
    1. Multi-Query Expansion: Genera variaciones de la pregunta para mejorar el 'Recall'.
    2. Cross-Encoder Reranking: Usa un modelo de IA especializado para reordenar los resultados
        con altísima precisión, reemplazando el conteo de palabras clave.
    """

    def __init__(self, llm, embedder, vector_store, 
                reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        super().__init__(llm, embedder, vector_store)
        # Cargamos el modelo de re-ranking (es ligero y rápido)
        # Nota: Esto se descarga la primera vez.
        print(f"🧠 [AdvancedRAG] Cargando modelo Cross-Encoder: {reranker_model}...")
        self.reranker = CrossEncoder(reranker_model)

    async def _generate_query_variations(self, query: str, n: int = 3) -> List[str]:
        """Genera formas alternativas de preguntar lo mismo para ampliar la búsqueda."""
        prompt = f"""Tu tarea es generar {n} versiones diferentes de la siguiente pregunta del usuario para recuperar documentos relevantes de una base de datos vectorial.
        Genera variaciones que usen sinónimos o enfoques alternativos.
        
        Pregunta original: "{query}"
        
        Salida (solo las preguntas separadas por saltos de línea):"""
        
        response = await self.llm.generate(prompt)
        # Limpieza básica
        variations = [v.strip("- ").strip() for v in response.split('\n') if v.strip()]
        return variations[:n]

    async def retrieve(self, query: str, k: int = 4, **kwargs) -> List[Chunk]:
        """
        Fase 1: Recuperación Ampliada (Recall)
        Fase 2: Reordenamiento Neuronal (Precision)
        """
        
        # --- PASO 1: Query Expansion ---
        print(f"🔍 [AdvancedRAG] Analizando: '{query}'")
        # Generamos variaciones (incluimos la original)
        queries = [query]
        variations = await self._generate_query_variations(query)
        queries.extend(variations)
        
        print(f"   -> Variaciones generadas: {variations}")

        # --- PASO 2: Búsqueda Vectorial Masiva ---
        # Buscamos para TODAS las variaciones y acumulamos resultados únicos.
        # Pedimos más candidatos (k*3) para darle margen al Reranker de filtrar.
        candidates_pool: Dict[str, Chunk] = {}
        
        # Lanzamos las búsquedas en paralelo
        tasks = [self.embedder.embed_query(q) for q in queries]
        query_vectors = await asyncio.gather(*tasks)
        
        search_tasks = [
            self.vector_store.search(vec, k=k*2) 
            for vec in query_vectors
        ]
        results_lists = await asyncio.gather(*search_tasks)
        
        # Deduplicación
        for res_list in results_lists:
            for chunk in res_list:
                if chunk.id not in candidates_pool:
                    candidates_pool[chunk.id] = chunk
        
        unique_candidates = list(candidates_pool.values())
        print(f"   -> Candidatos únicos recuperados: {len(unique_candidates)}")

        if not unique_candidates:
            return []

        # --- PASO 3: Cross-Encoder Reranking ---
        # El CrossEncoder necesita pares [ (Query, Doc1), (Query, Doc2), ... ]
        pairs = [[query, doc.content] for doc in unique_candidates]
        
        # Ejecutamos el modelo (es síncrono, así que usamos to_thread para no bloquear)
        scores = await asyncio.to_thread(self.reranker.predict, pairs)
        
        # Combinamos docs con sus scores y ordenamos
        scored_candidates = sorted(
            zip(unique_candidates, scores), 
            key=lambda x: x[1], 
            reverse=True
        )

        # Filtramos resultados con score muy bajo (opcional) y devolvemos Top K
        final_results = []
        print("   -> Ranking Final (Top Scores):")
        for doc, score in scored_candidates[:k]:
            print(f"      [{score:.4f}] {doc.content[:50]}...")
            final_results.append(doc)
            
        return final_results

    async def stream(self, query: str, k: int = 4, **kwargs) -> AsyncGenerator[str, None]:
        # Usamos el retrieve mejorado
        chunks = await self.retrieve(query, k=k)
        
        if not chunks:
            yield "No encontré información relevante tras un análisis exhaustivo."
            return

        context_text = self._build_context(chunks)
        
        prompt = f"""Responde a la pregunta usando el siguiente contexto verificado y reordenado por relevancia.

CONTEXTO:
{context_text}

PREGUNTA: {query}
RESPUESTA:"""
        
        async for token in self.llm.stream(prompt):
            yield token