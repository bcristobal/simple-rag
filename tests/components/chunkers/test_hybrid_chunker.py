import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock
from srag.core import Document, BaseEmbedder
from srag.components.chunkers.hybrid_chunker import HybridChunker

# --- Fixtures ---

@pytest.fixture
def mock_embedder():
    """
    Crea un Mock que cumple con la interfaz BaseEmbedder.
    Usamos AsyncMock porque los métodos de la interfaz son async.
    """
    embedder = MagicMock(spec=BaseEmbedder)
    embedder.embed_documents = AsyncMock()
    embedder.embed_query = AsyncMock()
    return embedder

@pytest.fixture
def chunker(mock_embedder):
    """Instancia el HybridChunker inyectando el mock."""
    return HybridChunker(
        embedder=mock_embedder,  # <--- Inyección de dependencia
        min_chunk_size_words=2,  # Pequeño para facilitar tests
        max_chunk_size_words=10,
        percentile_threshold=50.0
    )

# --- Tests ---

@pytest.mark.asyncio  # <--- Marca necesaria para tests asíncronos
async def test_markdown_splitting(chunker, mock_embedder):
    """
    Verifica que el chunker respete la estructura Markdown.
    Simulamos que el embedder devuelve vectores idénticos para que 
    la parte semántica NO divida nada, aislando la prueba de Markdown.
    """
    md_text = """
# Introducción
Texto de la intro.
## Sección 1
Detalle importante.
    """
    doc = Document(content=md_text, metadata={"source": "doc.md"})

    # Configuramos el mock para devolver vectores idénticos (similitud 1.0)
    # HybridChunker llama a embed_documents con una lista de oraciones.
    # Devolvemos vectores dummy de dimensión 2.
    mock_embedder.embed_documents.return_value = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]

    # Ejecutamos con AWAIT
    chunks = await chunker.split([doc])

    # Debería haber 2 chunks principales (Intro y Sección 1)
    # (El chunker semántico no dividió internamente porque los vectores eran iguales)
    assert len(chunks) == 2
    
    # Verificar contenido y metadatos
    assert "Texto de la intro" in chunks[0].content
    assert chunks[0].metadata["Header 1"] == "Introducción"
    
    assert "Detalle importante" in chunks[1].content
    assert chunks[1].metadata["Header 2"] == "Sección 1"

@pytest.mark.asyncio
async def test_semantic_splitting_logic(chunker, mock_embedder):
    """
    Verifica la lógica semántica simulando embeddings divergentes.
    """
    # 3 oraciones. 1 y 2 similares. 3 diferente.
    text = "Gato come. Gato duerme. Coche rápido."
    doc = Document(content=text)

    # Vectores simulados
    v1 = [1.0, 0.0] # Gato
    v2 = [1.0, 0.0] # Gato (Similitud 1.0 con anterior)
    v3 = [0.0, 1.0] # Coche (Similitud 0.0 con anterior)

    # El chunker llamará a embed_documents una vez con las 3 oraciones
    mock_embedder.embed_documents.return_value = [v1, v2, v3]

    chunks = await chunker.split([doc])

    # Esperamos 2 chunks:
    # 1. "Gato come. Gato duerme." (Agrupados por similitud)
    # 2. "Coche rápido." (Separado por diferencia)
    
    assert len(chunks) == 2
    assert "Gato come" in chunks[0].content
    assert "Gato duerme" in chunks[0].content
    assert "Coche rápido" in chunks[1].content

@pytest.mark.asyncio
async def test_fallback_no_markdown(chunker, mock_embedder):
    """Prueba texto plano sin headers."""
    text = "Oración uno. Oración dos."
    doc = Document(content=text)
    
    # Mock vectores iguales -> No split semántico
    mock_embedder.embed_documents.return_value = [[1,0], [1,0]]
    
    chunks = await chunker.split([doc])
    
    assert len(chunks) == 1
    assert chunks[0].content.strip() == text

@pytest.mark.asyncio
async def test_error_handling_embedder(chunker, mock_embedder):
    """Si el Embedder falla, debería devolver el chunk entero (fallback)."""
    doc = Document(content="Texto prueba. Otra cosa.")
    
    # Simular error en la llamada async
    mock_embedder.embed_documents.side_effect = Exception("API Error")
    
    chunks = await chunker.split([doc])
    
    # Debe retornar 1 chunk con todo el texto (graceful degradation)
    assert len(chunks) == 1
    assert "Texto prueba" in chunks[0].content