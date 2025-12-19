from srag.core import BaseChunker, Document, Chunk
from typing import List

class FixedLengthChunker(BaseChunker):
    """
    Chunker que divide documentos en fragmentos de longitud fija.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Inicializa el chunker con tamaño de chunk y solapamiento.

        :param chunk_size: Número máximo de caracteres por chunk.
        :param overlap: Número de caracteres que se solapan entre chunks consecutivos.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    async def split(self, documents: List[Document]) -> List[Chunk]:
        """
        Divide los documentos en chunks de longitud fija.

        :param documents: Lista de documentos a dividir.
        :return: Lista de chunks generados.
        """
        chunks: List[Chunk] = []
        for document in documents:
            text = document.content
            start = 0
            text_length = len(text)

            while start < text_length:
                end = min(start + self.chunk_size, text_length)
                chunk_text = text[start:end]

                # Mejor práctica: Copiar todos los metadatos y actualizar/añadir los nuevos
                chunk_metadata = document.metadata.copy()
                chunk_metadata.update({
                    "original_start": start,
                    "original_end": end
                })

                chunk = Chunk(
                    content=chunk_text,
                    metadata=chunk_metadata
                )
                chunks.append(chunk)

                start += self.chunk_size - self.overlap

        return chunks