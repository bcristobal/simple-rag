import re
import numpy as np
import ollama
from typing import List
from langchain_text_splitters import MarkdownHeaderTextSplitter
from srag.core import BaseChunker, Document, Chunk

class HybridChunker(BaseChunker):
    """
    Chunker Híbrido:
    1. Divide el documento por estructura Markdown (Encabezados).
    2. Divide el contenido de cada sección semánticamente usando embeddings.
    """

    def __init__(
        self, 
        ollama_host: str = 'http://127.0.0.1:11434',
        embedding_model: str = "nomic-embed-text",
        percentile_threshold: float = 90.0,
        min_chunk_size_words: int = 40,
        max_chunk_size_words: int = 300
    ):
        """
        Args:
            ollama_host: URL del servidor Ollama.
            embedding_model: Modelo de embeddings a usar para calcular similitud.
            percentile_threshold: Umbral para decidir cuándo romper un chunk semántico.
            min_chunk_size_words: Tamaño mínimo para intentar fusionar chunks pequeños.
            max_chunk_size_words: Tamaño máximo permitido.
        """
        self.client = ollama.Client(host=ollama_host)
        self.model = embedding_model
        self.percentile_threshold = percentile_threshold
        self.min_chunk_size_words = min_chunk_size_words
        self.max_chunk_size_words = max_chunk_size_words

        # Configuración de headers para Markdown
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]

    async def split(self, documents: List[Document]) -> List[Chunk]:
        """
        Implementación del método abstracto de BaseChunker.
        """
        final_chunks: List[Chunk] = []
        
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on)

        for doc in documents:
            # 1. División Estructural (Markdown)
            # LangChain devuelve sus propios objetos Document, no los nuestros
            try:
                md_splits = markdown_splitter.split_text(doc.content)
            except Exception:
                # Fallback si falla el split o no hay headers
                md_splits = []

            # Si no se encontraron headers, tratamos todo el texto como un bloque
            if not md_splits:
                # Creamos un objeto dummy compatible con la estructura de langchain
                from collections import namedtuple
                SimpleDoc = namedtuple('SimpleDoc', ['page_content', 'metadata'])
                md_splits = [SimpleDoc(page_content=doc.content, metadata={})]

            # 2. División Semántica por cada sección
            for md_section in md_splits:
                section_text = md_section.page_content
                section_meta = md_section.metadata # Headers extraídos (Header 1, Header 2...)

                # Aplicamos chunking semántico al texto de la sección
                semantic_texts = self._semantic_chunker(section_text)

                # 3. Crear objetos Chunk de SRAG
                for i, text_content in enumerate(semantic_texts):
                    # Combinamos metadatos originales + metadatos de headers
                    combined_metadata = doc.metadata.copy()
                    combined_metadata.update(section_meta)
                    combined_metadata["chunk_strategy"] = "hybrid"
                    
                    chunk = Chunk(
                        content=text_content,
                        document_id=doc.id,
                        metadata=combined_metadata,
                        index=len(final_chunks) # Índice global en la lista de retorno
                    )
                    final_chunks.append(chunk)

        return final_chunks

    # --- Métodos Internos (Lógica original adaptada) ---

    def _semantic_chunker(self, text: str) -> List[str]:
        """Lógica core de división semántica."""
        sentences = self._split_text_into_sentences(text)
        
        if len(sentences) < 2:
            return [text] if text.strip() else []

        # Obtener embeddings
        try:
            response = self.client.embed(model=self.model, input=sentences)
            embeddings = [np.array(e) for e in response['embeddings']]
        except Exception as e:
            print(f"⚠️ Error generando embeddings para chunking: {e}")
            return [text] # Fallback: devolver texto entero

        # Calcular distancias coseno
        adjacent_similarities = []
        for i in range(1, len(embeddings)):
            sim = self._cosine_similarity(embeddings[i], embeddings[i-1])
            adjacent_similarities.append(sim)
        
        if not adjacent_similarities:
            return [" ".join(sentences)]

        # Calcular umbral dinámico
        dynamic_threshold = np.percentile(adjacent_similarities, self.percentile_threshold)

        # Agrupar oraciones
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

        # Reparar chunks (merge de pequeños y split de grandes)
        return self._repair_chunks(initial_chunks)

    def _repair_chunks(self, chunks: List[str]) -> List[str]:
        """Lógica de fusión y división final por tamaño."""
        # 1. Merge de chunks pequeños
        merged_chunks = chunks[:]
        while True:
            if len(merged_chunks) <= 1:
                break
            
            did_merge = False
            repaired = []
            i = 0
            while i < len(merged_chunks):
                current = merged_chunks[i]
                
                # Si es pequeño y no es el último, intentamos unir con el siguiente
                if len(current.split()) < self.min_chunk_size_words and i < len(merged_chunks) - 1:
                    next_chunk = merged_chunks[i+1]
                    if len(current.split()) + len(next_chunk.split()) <= self.max_chunk_size_words:
                        repaired.append(current + " " + next_chunk)
                        i += 2
                        did_merge = True
                        continue
                
                repaired.append(current)
                i += 1
            
            merged_chunks = repaired
            if not did_merge:
                break

        # 2. Split de chunks demasiado grandes (Split forzado)
        final_chunks = []
        for chunk in merged_chunks:
            if len(chunk.split()) > self.max_chunk_size_words:
                # División simple a la mitad si excede el máximo
                sents = self._split_text_into_sentences(chunk)
                mid = len(sents) // 2
                part1 = " ".join(sents[:mid])
                part2 = " ".join(sents[mid:])
                if part1: final_chunks.append(part1)
                if part2: final_chunks.append(part2)
            else:
                final_chunks.append(chunk)
                
        return final_chunks

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