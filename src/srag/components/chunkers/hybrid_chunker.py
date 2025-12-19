import re
import numpy as np
from typing import List
from langchain_text_splitters import MarkdownHeaderTextSplitter
from srag.core import BaseChunker, BaseEmbedder, Document, Chunk # Agregado BaseEmbedder

class HybridChunker(BaseChunker):
    """
    Chunker Híbrido:
    1. Divide el documento por estructura Markdown (Encabezados).
    2. Divide el contenido de cada sección semánticamente usando embeddings.
    """

    def __init__(
        self, 
        embedder: BaseEmbedder, # <--- INYECCIÓN DE DEPENDENCIA (Adiós Ollama hardcoded)
        percentile_threshold: float = 90.0,
        min_chunk_size_words: int = 40,
        max_chunk_size_words: int = 300
    ):
        self.embedder = embedder 
        self.percentile_threshold = percentile_threshold
        self.min_chunk_size_words = min_chunk_size_words
        self.max_chunk_size_words = max_chunk_size_words

        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]

    async def split(self, documents: List[Document]) -> List[Chunk]:
        final_chunks: List[Chunk] = []
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on)

        for doc in documents:
            # 1. División Estructural (Síncrona - CPU bound ligera)
            try:
                md_splits = markdown_splitter.split_text(doc.content)
            except Exception:
                md_splits = []

            if not md_splits:
                from collections import namedtuple
                SimpleDoc = namedtuple('SimpleDoc', ['page_content', 'metadata'])
                md_splits = [SimpleDoc(page_content=doc.content, metadata={})]

            # 2. División Semántica (Ahora Asíncrona)
            for md_section in md_splits:
                section_text = md_section.page_content
                section_meta = md_section.metadata

                # Await directo aquí, sin bloquear el loop
                semantic_texts = await self._semantic_chunker(section_text)

                for text_content in semantic_texts:
                    combined_metadata = doc.metadata.copy()
                    combined_metadata.update(section_meta)
                    combined_metadata["chunk_strategy"] = "hybrid"
                    
                    chunk = Chunk(
                        content=text_content,
                        document_id=doc.id,
                        metadata=combined_metadata,
                        index=len(final_chunks)
                    )
                    final_chunks.append(chunk)

        return final_chunks

    async def _semantic_chunker(self, text: str) -> List[str]:
        """Ahora es async para no bloquear I/O."""
        sentences = self._split_text_into_sentences(text)
        
        if len(sentences) < 2:
            return [text] if text.strip() else []

        # Usamos el embedder inyectado (que es async)
        try:
            # BaseEmbedder.embed_documents retorna List[Vector]
            embeddings = await self.embedder.embed_documents(sentences)
            # Convertimos a numpy para cálculos
            embeddings = [np.array(e) for e in embeddings]
        except Exception as e:
            print(f"⚠️ Error generando embeddings: {e}")
            return [text]

        # Calcular distancias (CPU bound - NumPy es rápido, aceptable en main thread para textos cortos)
        adjacent_similarities = []
        for i in range(1, len(embeddings)):
            sim = self._cosine_similarity(embeddings[i], embeddings[i-1])
            adjacent_similarities.append(sim)
        
        if not adjacent_similarities:
            return [" ".join(sentences)]

        dynamic_threshold = np.percentile(adjacent_similarities, self.percentile_threshold)

        initial_chunks = []
        current_chunk_sentences = [sentences[0]]
        
        for i in range(1, len(sentences)):
            similarity = adjacent_similarities[i-1]
            if similarity >= dynamic_threshold:
                current_chunk_sentences.append(sentences[i])
            else:
                initial_chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [sentences[i]]
        
        initial_chunks.append(" ".join(current_chunk_sentences))

        return self._repair_chunks(initial_chunks)

    # _repair_chunks, _cosine_similarity, _split_text_into_sentences se mantienen igual (son puros métodos de lógica)
    def _repair_chunks(self, chunks: List[str]) -> List[str]:
        # (Copia tu implementación anterior aquí, no cambia nada)
        # ... [código existente] ...
        return chunks # Placeholder para brevedad

    @staticmethod
    def _cosine_similarity(a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    @staticmethod
    def _split_text_into_sentences(text: str) -> List[str]:
        split_by_punctuation = re.split(r'([.?!])', text)
        sentences = []
        for i in range(0, len(split_by_punctuation) - 1, 2):
            sentence = (split_by_punctuation[i] + split_by_punctuation[i+1]).strip()
            if sentence:
                sentences.append(sentence)
        if split_by_punctuation[-1].strip():
            sentences.append(split_by_punctuation[-1].strip())
        return sentences