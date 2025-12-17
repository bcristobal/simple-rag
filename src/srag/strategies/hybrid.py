from typing import List, AsyncGenerator, Dict, Set
from collections import defaultdict, Counter
import math
import re

from .base import BaseRAGStrategy
from srag.core import Chunk

class HybridRAG(BaseRAGStrategy):
    """
    Estrategia RAG Híbrida Funcional.
    
    Combina:
    1. Búsqueda Vectorial (Dense Retrieval) a través de ChromaDB.
    2. Búsqueda por Palabras Clave (Sparse Retrieval) simulada en memoria sobre los candidatos.
    3. Algoritmo de Fusión RRF (Reciprocal Rank Fusion) para el ranking final.
    """

    def _tokenize(self, text: str) -> Set[str]:
        """Tokenizador simple: minúsculas y elimina puntuación."""
        text = text.lower()
        # Mantenemos solo letras y números
        tokens = re.findall(r'\b\w+\b', text)
        # Filtramos palabras muy cortas (stopwords simples)
        return {t for t in tokens if len(t) > 2}

    def _compute_keyword_score(self, query: str, chunk: Chunk) -> float:
        """
        Calcula un score simple basado en frecuencia de términos (TF).
        Premia si el chunk contiene las palabras exactas de la query.
        """
        query_tokens = self._tokenize(query)
        chunk_tokens = self._tokenize(chunk.content)
        
        if not query_tokens:
            return 0.0
            
        score = 0.0
        for token in query_tokens:
            # Si la palabra clave está en el chunk, sumamos puntos
            if token in chunk_tokens:
                score += 1.0
        
        # Normalizamos por longitud para no favorecer solo textos largos excesivamente
        return score

    def _reciprocal_rank_fusion(self, vector_results: List[Chunk], keyword_results: List[Chunk], k=60) -> List[Chunk]:
        """
        Algoritmo estándar RRF.
        Score = 1 / (constante + rango).
        """
        fused_scores = defaultdict(float)
        doc_map = {}

        # 1. Procesar ranking vectorial
        for rank, doc in enumerate(vector_results):
            doc_map[doc.id] = doc
            fused_scores[doc.id] += 1 / (k + rank + 1)

        # 2. Procesar ranking por palabras clave
        for rank, doc in enumerate(keyword_results):
            doc_map[doc.id] = doc
            fused_scores[doc.id] += 1 / (k + rank + 1)

        # 3. Ordenar por score final descendente
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        
        return [doc_map[doc_id] for doc_id in sorted_ids]

    async def retrieve(self, query: str, k: int = 4, **kwargs) -> List[Chunk]:
        """
        Recuperación Híbrida:
        Trae un 'pool' grande de vectores y lo reordena por palabras clave.
        """
        # A. Recuperamos un 'pool' más grande de candidatos por vector (ej. k * 3)
        # Esto aumenta la probabilidad de traer documentos que tengan buenas palabras clave
        # aunque su vector no sea perfecto.
        pool_size = k * 3
        
        query_vector = await self.embedder.embed_query(query)
        candidates = await self.vector_store.search(query_vector, k=pool_size)
        
        # La lista 'candidates' ya es nuestro Ranking Vectorial (List A)
        vector_ranked = candidates[:]

        # B. Generamos el Ranking por Palabras Clave (List B)
        # Calculamos score léxico para cada candidato
        scored_candidates = []
        for doc in candidates:
            score = self._compute_keyword_score(query, doc)
            scored_candidates.append((doc, score))
        
        # Ordenamos por score de palabra clave
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        keyword_ranked = [item[0] for item in scored_candidates]

        # C. Fusionamos con RRF
        final_results = self._reciprocal_rank_fusion(vector_ranked, keyword_ranked)

        # Devolvemos solo los top K solicitados
        return final_results[:k]

    async def stream(self, query: str, k: int = 4, **kwargs) -> AsyncGenerator[str, None]:
        # 1. Retrieve Híbrido
        chunks = await self.retrieve(query, k=k)
        
        if not chunks:
            yield "No se encontró información relevante."
            return

        # 2. Construir contexto
        context_text = self._build_context(chunks)

        # 3. Prompt específico para Hybrid
        prompt = f"""Actúa como un analista experto. Usa el siguiente contexto (obtenido mediante búsqueda híbrida) para responder.

CONTEXTO:
{context_text}

PREGUNTA DEL USUARIO: 
{query}

RESPUESTA:"""

        # 4. Generar
        async for token in self.llm.stream(prompt):
            yield token