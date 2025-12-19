import pytest
import asyncio
import numpy as np
from typing import List

# Importamos las piezas reales
from srag.core import Document, BaseLoader, BaseEmbedder, Chunk
from srag.components.chunkers.fixed_length_chunker import FixedLengthChunker
from srag.components.vectorstores.chroma_vectorstore import ChromaVectorStore
from srag.pipeline.ingestion import IngestionPipeline

# --- MOCKS (Simuladores para no depender de APIs externas) ---

class MockLoader(BaseLoader):
    """Simula cargar un archivo con texto conocido."""
    async def load(self, url_file: List[str] = None) -> List[Document]:
        return [
            Document(
                content="Python es un lenguaje de programación versátil. " * 10, # Texto repetido para tener volumen
                metadata={"source": "test_doc.txt", "author": "Tester"}
            )
        ]

class MockEmbedder(BaseEmbedder):
    """
    Genera vectores deterministas sin llamar a una IA real.
    Devuelve siempre vectores de dimensión 3 para testear.
    """
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Genera un vector "dummy" basado en la longitud del texto para que no sean todos iguales
        return [[len(t) * 0.1, 0.5, 0.9] for t in texts]

    async def embed_query(self, text: str) -> List[float]:
        return [0.1, 0.5, 0.9]

# --- TESTS DE INTEGRACIÓN ---

@pytest.mark.asyncio
async def test_full_ingestion_pipeline_and_retrieval():
    """
    Prueba End-to-End:
    1. Pipeline de Ingesta (Load -> Chunk -> Embed -> Store)
    2. Verificación de almacenamiento en ChromaDB
    3. Recuperación (Retrieval)
    """
    
    # 1. Configuración de Componentes
    loader = MockLoader()
    
    # Usamos el Chunker real (FixedLength)
    chunker = FixedLengthChunker(chunk_size=50, overlap=10)
    
    embedder = MockEmbedder()
    
    # Chroma en memoria (sin path ni host) para que se borre al terminar el test
    vector_store = ChromaVectorStore(collection_name="test_integration_collection")

    # Instanciamos el Pipeline
    pipeline = IngestionPipeline(loader, chunker, embedder, vector_store)

    # 2. Ejecutar Ingesta (Black Box)
    await pipeline.run(batch_size=2) # Batch pequeño para probar la lógica de lotes

    # 3. Verificaciones Post-Ingesta
    
    # A. Comprobar que ChromaDB tiene datos
    # Accedemos al cliente interno de Chroma para contar (esto es específico de Chroma)
    count = vector_store.collection.count()
    assert count > 0, "La colección de Chroma debería tener chunks indexados."
    print(f"\n✅ Se indexaron {count} chunks correctamente.")

    # B. Prueba de Recuperación (Retrieval)
    # Simulamos una búsqueda
    query_vec = await embedder.embed_query("query de prueba")
    
    results = await vector_store.search(query_vec, k=2)
    
    assert len(results) == 2, "Debería recuperar 2 resultados."
    assert isinstance(results[0], Chunk), "Los resultados deben ser objetos Chunk."
    
    # Verificar que los metadatos se preservaron a través de todo el flujo
    assert results[0].metadata["source"] == "test_doc.txt"
    assert "original_start" in results[0].metadata
    
    print("✅ Recuperación exitosa y metadatos verificados.")

    # 4. Limpieza (Opcional, Chroma en memoria muere con el proceso, pero es buena práctica borrar)
    # await vector_store.delete([c.id for c in results]) 

@pytest.mark.asyncio
async def test_pipeline_handles_empty_loader():
    """Verifica que el pipeline no explote si no hay documentos."""
    
    class EmptyLoader(BaseLoader):
        async def load(self, *args): return []

    pipeline = IngestionPipeline(
        loader=EmptyLoader(),
        chunker=FixedLengthChunker(),
        embedder=MockEmbedder(),
        vector_store=ChromaVectorStore(collection_name="empty_test")
    )
    
    # No debería lanzar error
    await pipeline.run()
    assert True