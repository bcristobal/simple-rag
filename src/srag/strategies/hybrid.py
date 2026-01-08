from typing import List, AsyncGenerator, Dict, Set, Optional
from collections import defaultdict
import re
import unicodedata

from .base import BaseRAGStrategy
from srag.core import Chunk

class HybridRAG(BaseRAGStrategy):
    """
    Estrategia RAG Híbrida Mejorada.
    
    Combina:
    1. Búsqueda Vectorial (Dense Retrieval) con alto recall (pool grande).
    2. Filtrado por Palabras Clave (Sparse Retrieval) con normalización lingüística.
    3. Algoritmo de Fusión RRF (Reciprocal Rank Fusion).
    """

    def _normalize_text(self, text: str) -> str:
        """
        Normaliza el texto para mejorar el matching en español:
        1. Elimina acentos/diacríticos (NFD normalization).
        2. Convierte a minúsculas.
        """
        if not text:
            return ""
        # Descompone caracteres (ej. 'ó' -> 'o' + '´') y elimina los de categoría 'Mn' (Non-spacing Mark)
        return ''.join(
            c for c in unicodedata.normalize('NFD', text) 
            if unicodedata.category(c) != 'Mn'
        ).lower()

    def _tokenize(self, text: str) -> Set[str]:
        """Tokenizador normalizado."""
        # 1. Normalizamos (quitamos tildes, ñ -> n, etc. para búsqueda robusta)
        clean_text = self._normalize_text(text)
        
        # 2. Extraemos palabras (alfanuméricas)
        tokens = re.findall(r'\b\w+\b', clean_text)
        
        # 3. Filtramos stopwords muy cortas
        return {t for t in tokens if len(t) > 2}

    def _compute_keyword_score(self, query: str, chunk: Chunk) -> float:
        """Calcula score basado en coincidencia de tokens normalizados."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0.0
            
        chunk_tokens = self._tokenize(chunk.content)
        
        score = 0.0
        matches = 0
        for token in query_tokens:
            if token in chunk_tokens:
                score += 1.0
                matches += 1
        
        # (Opcional) Boost si hay muchas coincidencias
        return score

    def _reciprocal_rank_fusion(self, vector_results: List[Chunk], keyword_results: List[Chunk], k=60) -> List[Chunk]:
        """Algoritmo estándar RRF."""
        fused_scores = defaultdict(float)
        doc_map = {}

        # 1. Ranking Vectorial
        for rank, doc in enumerate(vector_results):
            doc_map[doc.id] = doc
            fused_scores[doc.id] += 1 / (k + rank + 1)

        # 2. Ranking Keywords
        for rank, doc in enumerate(keyword_results):
            doc_map[doc.id] = doc
            fused_scores[doc.id] += 1 / (k + rank + 1)

        # 3. Ordenar final
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_ids]

    async def retrieve(
        self, 
        query: str, 
        k: int = 4, 
        vector_query: Optional[str] = None, 
        **kwargs
    ) -> List[Chunk]:
        """
        Recuperación Híbrida (Vector + Keywords).

        Args:
            query: La pregunta original del usuario (usada para keywords).
            k: Número de resultados finales.
            vector_query: (Opcional) Texto alternativo para la búsqueda vectorial 
                          (ej. documento 'HyDE' alucinado). Si no se da, usa 'query'.
        """
        # A. Configuración de Búsqueda Vectorial
        # Usamos vector_query si existe (para Modular/HyDE), sino la query original
        search_text = vector_query if vector_query else query
        
        # Aumentamos el pool significativamente (k * 10) para emular un "Recall" alto
        # y permitir que el re-ranking de palabras clave tenga material con el que trabajar.
        pool_size = max(k * 10, 50) 
        
        # 1. Búsqueda Vectorial (Dense)
        q_vec = await self.embedder.embed_query(search_text)
        candidates = await self.vector_store.search(q_vec, k=pool_size)
        
        if not candidates:
            return []

        # Lista A: Ranking puramente vectorial (recortado a un tamaño razonable para RRF)
        # Tomamos los top K*2 para la fusión vectorial
        vector_ranked = candidates[:k*2]

        # 2. Búsqueda por Palabras Clave (Sparse / Re-ranking)
        # IMPORTANTE: Usamos siempre la 'query' original, nunca la alucinación HyDE
        scored_candidates = []
        for doc in candidates:
            score = self._compute_keyword_score(query, doc)
            scored_candidates.append((doc, score))
        
        # Ordenamos por score de keywords
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Lista B: Los mejores candidatos según palabras clave
        # Filtramos aquellos que tengan score > 0 para no introducir ruido
        keyword_ranked = [item[0] for item in scored_candidates if item[1] > 0]
        
        # Si no hubo coincidencias de palabras clave, usamos el ranking vectorial como fallback
        if not keyword_ranked:
            keyword_ranked = vector_ranked

        # 3. Fusión RRF
        final_results = self._reciprocal_rank_fusion(vector_ranked, keyword_ranked)

        return final_results[:k]

    async def stream(self, query: str, k: int = 4, **kwargs) -> AsyncGenerator[str, None]:
        chunks = await self.retrieve(query, k=k, **kwargs)
        
        if not chunks:
            yield "No se encontró información relevante."
            return

        context_text = self._build_context(chunks)
        
        prompt = f"""Actúa como un analista experto. Usa el siguiente contexto (obtenido mediante búsqueda híbrida) para responder.

CONTEXTO:
{context_text}

PREGUNTA DEL USUARIO: 
{query}

RESPUESTA:"""

        async for token in self.llm.stream(prompt):
            yield token