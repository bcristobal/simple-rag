import logging
import re
import numpy as np
from typing import List, Tuple, Dict
from collections import namedtuple
from srag.core import BaseChunker, BaseEmbedder, Document, Chunk

# Definimos una estructura simple para manejar los splits internos
SimpleDoc = namedtuple('SimpleDoc', ['page_content', 'metadata'])

# 1. Configuración del Logger
logger = logging.getLogger(__name__)

class HybridChunker(BaseChunker):
    """
    Chunker Híbrido (Sin dependencias externas de LangChain):
    1. Divide el documento por estructura Markdown (Encabezados) de forma nativa.
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

        # Ordenados para mantener consistencia en la jerarquía
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        
        logger.debug(
            f"HybridChunker inicializado | Threshold: {percentile_threshold}% | "
            f"Min/Max Words: {min_chunk_size_words}/{max_chunk_size_words}"
        )

    async def split(self, documents: List[Document]) -> List[Chunk]:
        logger.info(f"Iniciando división híbrida (Nativa) para {len(documents)} documentos...")
        
        final_chunks: List[Chunk] = []

        for i, doc in enumerate(documents):
            doc_id = doc.metadata.get('source', f"doc_{i}")
            
            # 1. División Estructural (Nativa)
            try:
                md_splits = self._split_markdown(doc.content)
                logger.debug(f"[{doc_id}] Estructura detectada: {len(md_splits)} secciones Markdown.")
            except Exception as e:
                logger.warning(f"[{doc_id}] Falló Markdown splitting nativo. Usando texto plano. Error: {e}")
                md_splits = [SimpleDoc(page_content=doc.content, metadata={})]

            if not md_splits:
                md_splits = [SimpleDoc(page_content=doc.content, metadata={})]

            # 2. División Semántica
            chunks_generated_for_doc = 0
            
            for section_idx, md_section in enumerate(md_splits):
                section_text = md_section.page_content
                section_meta = md_section.metadata

                # Llamada al chunker semántico
                semantic_texts = await self._semantic_chunker(section_text, context_id=f"{doc_id}_s{section_idx}")

                for text_content in semantic_texts:
                    combined_metadata = doc.metadata.copy()
                    combined_metadata.update(section_meta)
                    combined_metadata["chunk_strategy"] = "hybrid_v2"
                    
                    chunk = Chunk(
                        content=text_content,
                        # Usamos doc.id si existe, si no un hash o el doc_id generado
                        metadata=combined_metadata
                    )
                    # Nota: Si tu clase Chunk espera 'document_id' explícito en __init__, ajusta esta línea:
                    # document_id=doc.id 
                    
                    final_chunks.append(chunk)
                    chunks_generated_for_doc += 1
            
            logger.info(f"[{doc_id}] Procesado completado. Generados {chunks_generated_for_doc} chunks.")

        return final_chunks

    def _split_markdown(self, text: str) -> List[SimpleDoc]:
        """
        Divide texto Markdown por encabezados respetando la jerarquía.
        Reemplaza a MarkdownHeaderTextSplitter de LangChain.
        """
        lines = text.split('\n')
        headers_map = {h[0]: h[1] for h in self.headers_to_split_on}
        
        splits = []
        current_content = []
        current_metadata = {}
        in_code_block = False
        
        for line in lines:
            stripped_line = line.strip()
            
            # Detectar inicio/fin de bloque de código para no romper headers dentro de código
            if stripped_line.startswith("```"):
                in_code_block = not in_code_block
                current_content.append(line)
                continue
            
            if in_code_block:
                current_content.append(line)
                continue
                
            # Detectar encabezados (#, ##, etc.)
            header_match = re.match(r'^(#{1,6})\s+(.*)', line)
            if header_match:
                hashes, title = header_match.groups()
                
                # Solo dividimos si el nivel de header está en nuestra configuración
                if hashes in headers_map:
                    # 1. Guardar lo acumulado hasta ahora
                    if current_content:
                        content_str = "\n".join(current_content).strip()
                        if content_str:
                            splits.append(SimpleDoc(page_content=content_str, metadata=current_metadata.copy()))
                        current_content = []
                    
                    # 2. Actualizar Jerarquía de Metadatos
                    header_key = headers_map[hashes]
                    current_level = len(hashes)
                    
                    # Lógica: Mantenemos padres (niveles menores), borramos hermanos/hijos anteriores
                    new_metadata = {}
                    for h_hashes, h_name in self.headers_to_split_on:
                        h_level = len(h_hashes)
                        if h_level < current_level:
                            if h_name in current_metadata:
                                new_metadata[h_name] = current_metadata[h_name]
                    
                    new_metadata[header_key] = title.strip()
                    current_metadata = new_metadata
                else:
                    # Es un header (ej #####) que no configuramos para split, lo tratamos como texto
                    current_content.append(line)
            else:
                current_content.append(line)
        
        # Flush final
        if current_content:
            content_str = "\n".join(current_content).strip()
            if content_str:
                splits.append(SimpleDoc(page_content=content_str, metadata=current_metadata.copy()))
                
        return splits

    async def _semantic_chunker(self, text: str, context_id: str = "unknown") -> List[str]:
        """Divide texto basado en similitud semántica (Sin cambios)."""
        sentences = self._split_text_into_sentences(text)
        
        if len(sentences) < 2:
            return [text] if text.strip() else []

        try:
            embeddings = await self.embedder.embed_documents(sentences)
            embeddings = [np.array(e) for e in embeddings]
        except Exception as e:
            logger.error(f"[{context_id}] ⚠️ Fallo crítico en Embedder: {e}. Retornando bloque sin dividir.")
            return [text]

        adjacent_similarities = []
        for i in range(1, len(embeddings)):
            sim = self._cosine_similarity(embeddings[i], embeddings[i-1])
            adjacent_similarities.append(sim)
        
        if not adjacent_similarities:
            return [" ".join(sentences)]

        # Calcula umbral dinámico
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

    def _repair_chunks(self, chunks: List[str]) -> List[str]:
        """Fusiona chunks pequeños."""
        if not chunks: return []

        repaired_chunks = []
        current_buffer = ""

        for chunk in chunks:
            proposal = (current_buffer + " " + chunk).strip() if current_buffer else chunk
            # Conteo simple de palabras
            word_count = len(proposal.split())
            
            if word_count >= self.min_chunk_size_words:
                repaired_chunks.append(proposal)
                current_buffer = "" 
            else:
                current_buffer = proposal
        
        if current_buffer:
            if repaired_chunks:
                last_chunk = repaired_chunks.pop()
                merged_last = (last_chunk + " " + current_buffer).strip()
                repaired_chunks.append(merged_last)
            else:
                repaired_chunks.append(current_buffer)
                
        return repaired_chunks

    @staticmethod
    def _cosine_similarity(a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0: return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    @staticmethod
    def _split_text_into_sentences(text: str) -> List[str]:
        # Regex simple para oraciones (se podría mejorar para abreviaturas, etc.)
        split_by_punctuation = re.split(r'([.?!])', text)
        sentences = []
        for i in range(0, len(split_by_punctuation) - 1, 2):
            sentence = (split_by_punctuation[i] + split_by_punctuation[i+1]).strip()
            if sentence:
                sentences.append(sentence)
        if split_by_punctuation[-1].strip():
            sentences.append(split_by_punctuation[-1].strip())
        return sentences