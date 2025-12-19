import logging
from typing import List, Generator, Iterable
from itertools import islice
from srag.core import BaseLoader, BaseChunker, BaseEmbedder, BaseVectorStore

# Configuración básica de logging (el usuario final debería configurarlo, pero esto es mejor que print)
logger = logging.getLogger(__name__)

def batched(iterable: Iterable, n: int) -> Generator[tuple, None, None]:
    """Helper para procesar datos en lotes y ahorrar memoria."""
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

class IngestionPipeline:
    def __init__(
        self, 
        loader: BaseLoader, 
        chunker: BaseChunker, 
        embedder: BaseEmbedder, 
        vector_store: BaseVectorStore
    ):
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    async def run(self, batch_size: int = 50):
        """
        Ejecuta el flujo completo de ingesta con soporte de batching.
        Args:
            batch_size: Número de chunks a procesar y guardar a la vez.
        """
        logger.info("📥 [Pipeline] Cargando documentos...")
        raw_docs = await self.loader.load()
        
        if not raw_docs:
            logger.warning("⚠️ [Pipeline] No se encontraron documentos.")
            return

        logger.info(f"✂️ [Pipeline] Dividiendo {len(raw_docs)} documentos...")
        
        # CORRECCIÓN: Llamada directa con await. Ya es async real gracias al cambio en HybridChunker.
        # No usamos run_in_executor porque chunker.split es una corrutina.
        chunks = await self.chunker.split(raw_docs)
        
        total_chunks = len(chunks)
        logger.info(f"🧠 [Pipeline] Procesando {total_chunks} chunks en lotes de {batch_size}...")

        # CORRECCIÓN: Batching para evitar OOM (Out Of Memory)
        indexed_count = 0
        for batch_idx, batch_chunks in enumerate(batched(chunks, batch_size)):
            batch_list = list(batch_chunks) # batched devuelve tuplas
            
            # 1. Embed (Solo del lote actual)
            textos = [c.content for c in batch_list]
            vectores = await self.embedder.embed_documents(textos)
            
            # 2. Asignar
            for chunk, vector in zip(batch_list, vectores):
                chunk.embedding = vector
            
            # 3. Store (Solo del lote actual)
            await self.vector_store.add(batch_list)
            
            indexed_count += len(batch_list)
            logger.info(f"   💾 Guardado lote {batch_idx + 1} ({indexed_count}/{total_chunks})")

        logger.info("✅ [Pipeline] Ingesta completada con éxito.")