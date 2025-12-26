import logging
import re
import numpy as np
from typing import List, Tuple
from collections import namedtuple
from langchain_text_splitters import MarkdownHeaderTextSplitter
from srag.core import BaseChunker, BaseEmbedder, Document, Chunk

# 1. Configuración del Logger
logger = logging.getLogger(__name__)

class HybridChunker(BaseChunker):
    """
    Chunker Híbrido:
    1. Divide el documento por estructura Markdown (Encabezados).
    2. Divide el contenido de cada sección semánticamente usando embeddings.
    """

    def __init__(
        self, 
        embedder: BaseEmbedder, 
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
        
        # DEBUG: Configuración inicial. Útil para verificar si los parámetros inyectados son correctos.
        logger.debug(
            f"HybridChunker inicializado | Threshold: {percentile_threshold}% | "
            f"Min/Max Words: {min_chunk_size_words}/{max_chunk_size_words}"
        )

    async def split(self, documents: List[Document]) -> List[Chunk]:
        # INFO: Inicio del proceso macro.
        logger.info(f"Iniciando división híbrida para {len(documents)} documentos...")
        
        final_chunks: List[Chunk] = []
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on)

        for i, doc in enumerate(documents):
            doc_id = doc.metadata.get('source', f"doc_{i}")
            
            # 1. División Estructural
            try:
                md_splits = markdown_splitter.split_text(doc.content)
                # DEBUG: Ver cuántas secciones detectó el splitter de Markdown
                logger.debug(f"[{doc_id}] Estructura detectada: {len(md_splits)} secciones Markdown.")
            except Exception as e:
                # WARNING: Si falla el parser de Markdown, no detenemos todo, pero avisamos.
                logger.warning(f"[{doc_id}] Falló Markdown splitting. Usando texto plano. Error: {e}")
                md_splits = []

            # Fallback si no hay estructura Markdown o falló el split
            if not md_splits:
                SimpleDoc = namedtuple('SimpleDoc', ['page_content', 'metadata'])
                md_splits = [SimpleDoc(page_content=doc.content, metadata={})]

            # 2. División Semántica
            chunks_generated_for_doc = 0
            
            for section_idx, md_section in enumerate(md_splits):
                section_text = md_section.page_content
                section_meta = md_section.metadata

                # DEBUG: Trazabilidad fina. Solo se ve si activas modo debug profundo.
                # logger.debug(f"[{doc_id}] Procesando sección {section_idx+1}/{len(md_splits)} ({len(section_text)} chars)")

                semantic_texts = await self._semantic_chunker(section_text, context_id=f"{doc_id}_s{section_idx}")

                for text_content in semantic_texts:
                    combined_metadata = doc.metadata.copy()
                    combined_metadata.update(section_meta)
                    combined_metadata["chunk_strategy"] = "hybrid"
                    
                    chunk = Chunk(
                        content=text_content,
                        document_id=doc.id, # Asumiendo que doc tiene .id, sino usar doc_id generado
                        metadata=combined_metadata,
                        index=len(final_chunks)
                    )
                    final_chunks.append(chunk)
                    chunks_generated_for_doc += 1
            
            # INFO: Resumen por documento
            logger.info(f"[{doc_id}] Procesado completado. Generados {chunks_generated_for_doc} chunks.")

        return final_chunks

    async def _semantic_chunker(self, text: str, context_id: str = "unknown") -> List[str]:
        """Divide texto basado en similitud semántica."""
        sentences = self._split_text_into_sentences(text)
        
        if len(sentences) < 2:
            return [text] if text.strip() else []

        try:
            # DEBUG: Aquí ocurre la llamada costosa (API o GPU).
            # logger.debug(f"[{context_id}] Generando embeddings para {len(sentences)} oraciones...")
            
            embeddings = await self.embedder.embed_documents(sentences)
            embeddings = [np.array(e) for e in embeddings]
        except Exception as e:
            # ERROR: Esto es grave porque degrada la calidad a "texto plano".
            logger.error(f"[{context_id}] ⚠️ Fallo crítico en Embedder: {e}. Retornando bloque sin dividir.")
            return [text]

        adjacent_similarities = []
        for i in range(1, len(embeddings)):
            sim = self._cosine_similarity(embeddings[i], embeddings[i-1])
            adjacent_similarities.append(sim)
        
        if not adjacent_similarities:
            return [" ".join(sentences)]

        dynamic_threshold = np.percentile(adjacent_similarities, self.percentile_threshold)
        
        # DEBUG: Ver el umbral ayuda a tunear el parámetro 'percentile_threshold'
        # logger.debug(f"[{context_id}] Umbral semántico calculado: {dynamic_threshold:.4f}")

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

        repaired = self._repair_chunks(initial_chunks)
        
        # DEBUG: Ver si la reparación unió muchos chunks pequeños
        if len(repaired) != len(initial_chunks):
            logger.debug(f"[{context_id}] Repair: {len(initial_chunks)} -> {len(repaired)} chunks.")
            
        return repaired

    def _repair_chunks(self, chunks: List[str]) -> List[str]:
        """
        Fusiona chunks adyacentes que son demasiado pequeños (menores a min_chunk_size_words).
        Esto evita tener fragmentos con muy poco contexto semántico.
        """
        if not chunks:
            return []

        repaired_chunks = []
        current_buffer = ""
        chunks_merged_count = 0

        for chunk in chunks:
            # Unimos el buffer acumulado (pequeño) con el chunk actual
            proposal = (current_buffer + " " + chunk).strip() if current_buffer else chunk
            
            # Contamos palabras (aproximación rápida por espacios)
            word_count = len(proposal.split())
            
            if word_count >= self.min_chunk_size_words:
                # Si cumple el tamaño mínimo, lo aceptamos como chunk válido
                repaired_chunks.append(proposal)
                current_buffer = "" # Reiniciamos el buffer
            else:
                # Si es muy pequeño, lo guardamos en el buffer para unirlo al siguiente
                # DEBUG: Solo descomentar si necesitas ver qué frases se están uniendo
                # logger.debug(f"Chunk pequeño detectado ({word_count} palabras). Acumulando...")
                current_buffer = proposal
                chunks_merged_count += 1
        
        # Gestionar el "residuo" final (lo que quedó en current_buffer al terminar el bucle)
        if current_buffer:
            if repaired_chunks:
                # Estrategia: Unir al último chunk válido para no dejar un chunk "enano" al final.
                # Esto asegura que el último fragmento siempre tenga contexto suficiente.
                last_chunk = repaired_chunks.pop()
                merged_last = (last_chunk + " " + current_buffer).strip()
                repaired_chunks.append(merged_last)
                chunks_merged_count += 1
            else:
                # Caso borde: Todo el documento era más pequeño que min_chunk_size_words
                # No queda otra que devolverlo tal cual.
                repaired_chunks.append(current_buffer)

        # DEBUG: Información útil para saber cuánto se compactó el texto
        if chunks_merged_count > 0:
            logger.debug(
                f"Repair finished: {len(chunks)} chunks originales -> {len(repaired_chunks)} finales. "
                f"(Se fusionaron {chunks_merged_count} veces)"
            )
                
        return repaired_chunks

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