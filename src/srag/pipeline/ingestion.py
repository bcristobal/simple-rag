from typing import List
from srag.core import BaseLoader, BaseChunker, BaseEmbedder, BaseVectorStore
import asyncio

class IngestionPipeline:
    """
    Orquestador para el proceso de indexación de documentos:
    Load -> Split -> Embed -> Store
    """
    
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

    async def run(self):
        """Ejecuta el flujo completo de ingesta."""
        print("📥 [Pipeline] Cargando documentos...")
        raw_docs = await self.loader.load()
        
        if not raw_docs:
            print("⚠️ [Pipeline] No se encontraron documentos.")
            return

        print(f"✂️ [Pipeline] Dividiendo {len(raw_docs)} documentos...")
        loop = asyncio.get_running_loop()
        chunks = await loop.run_in_executor(None, self.chunker.split, raw_docs)
        
        print(f"🧠 [Pipeline] Generando embeddings para {len(chunks)} chunks...")
        # Extraemos texto, generamos vectores
        textos = [c.content for c in chunks]
        vectores = await self.embedder.embed_documents(textos)
        
        # Asignamos vectores a los objetos Chunk
        for chunk, vector in zip(chunks, vectores):
            chunk.embedding = vector
            
        print(f"💾 [Pipeline] Guardando en VectorStore...")
        await self.vector_store.add(chunks)
        print("✅ [Pipeline] Ingesta completada con éxito.")