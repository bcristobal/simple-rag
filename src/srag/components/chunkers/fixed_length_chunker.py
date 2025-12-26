import logging
from srag.core import BaseChunker, Document, Chunk
from typing import List

# 1. Instanciamos el logger del módulo
logger = logging.getLogger(__name__)

class FixedLengthChunker(BaseChunker):
    """
    Chunker que divide documentos en fragmentos de longitud fija.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Inicializa el chunker con tamaño de chunk y solapamiento.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # DEBUG: Útil para verificar la configuración al arrancar la app, 
        # pero no ensucia el log normal.
        logger.debug(f"FixedLengthChunker inicializado (size={chunk_size}, overlap={overlap})")

    async def split(self, documents: List[Document]) -> List[Chunk]:
        """
        Divide los documentos en chunks de longitud fija.
        """
        # INFO: Queremos saber cuándo empieza un proceso pesado.
        logger.info(f"Iniciando división de {len(documents)} documentos...")
        
        chunks: List[Chunk] = []
        
        for i, document in enumerate(documents):
            text = document.content
            
            # WARNING: Detectar datos de mala calidad es vital en producción.
            if not text:
                source = document.metadata.get('source', f'doc_index_{i}')
                logger.warning(f"Documento vacío detectado: {source}. Se omitirá.")
                continue

            # DEBUG: Si algo falla, queremos saber en qué documento específico fue.
            # Usamos metadata.get('source') o un identificador si existe, sino el índice.
            doc_id = document.metadata.get('file_path', document.metadata.get('source', f"doc_{i}"))
            logger.debug(f"Procesando documento: '{doc_id}' (Longitud: {len(text)} caracteres)")

            start = 0
            text_length = len(text)

            while start < text_length:
                end = min(start + self.chunk_size, text_length)
                chunk_text = text[start:end]

                chunk_metadata = document.metadata.copy()
                chunk_metadata.update({
                    "original_start": start,
                    "original_end": end,
                    # Opcional: Añadir el ID del chunk relativo al doc puede ser útil
                    "chunk_index": len(chunks) 
                })

                chunk = Chunk(
                    content=chunk_text,
                    metadata=chunk_metadata
                )
                chunks.append(chunk)

                start += self.chunk_size - self.overlap
        
        # INFO: Resumen final. Esto te permite calcular el "ratio de expansión"
        # (ej. 1 documento -> 50 chunks).
        logger.info(f"División completada. Generados {len(chunks)} chunks a partir de {len(documents)} documentos.")
        
        return chunks