import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from srag.core import Document
from  srag.components.chunkers.hybrid_chunker import HybridChunker

# --- Fixtures ---

@pytest.fixture
def mock_ollama_client():
    """Mockea el cliente de Ollama."""
    with patch('srag.components.chunkers.hybrid_chunker.ollama.Client') as MockClient:
        mock_instance = MockClient.return_value
        yield mock_instance

@pytest.fixture
def chunker(mock_ollama_client):
    return HybridChunker(
        ollama_host="http://fake",
        min_chunk_size_words=2, # Pequeño para facilitar tests
        max_chunk_size_words=10
    )

# --- Tests ---

def test_markdown_splitting(chunker, mock_ollama_client):
    """
    Verifica que el chunker respete la estructura Markdown.
    Simulamos que el 'semantic split' no divide nada (retorna el texto entero)
    para aislar la prueba de la parte Markdown.
    """
    # Texto con estructura Markdown
    md_text = """
# Introducción
Texto de la intro.
## Sección 1
Detalle importante.
    """
    doc = Document(content=md_text, metadata={"source": "doc.md"})

    # Simulamos que embed no falla, pero hacemos un bypass del semantic split
    # Mockeamos _semantic_chunker para que devuelva el texto tal cual
    # así probamos solo la división por headers.
    with patch.object(chunker, '_semantic_chunker', side_effect=lambda x: [x.strip()]):
        chunks = chunker.split([doc])

    # Debería haber 2 chunks (Intro y Sección 1)
    assert len(chunks) == 2
    
    # Verificar metadatos de headers
    assert chunks[0].content == "Texto de la intro."
    assert chunks[0].metadata["Header 1"] == "Introducción"
    
    assert chunks[1].content == "Detalle importante."
    assert chunks[1].metadata["Header 2"] == "Sección 1"

def test_semantic_splitting_logic(chunker, mock_ollama_client):
    """
    Verifica la lógica semántica simulando embeddings.
    Creamos 3 oraciones. Haremos que la 1 y 2 sean muy similares, 
    y la 3 muy diferente, para forzar un corte.
    """
    text = "Gato come. Gato duerme. Coche rápido."
    doc = Document(content=text)

    # Vectores simulados
    # v1 y v2 idénticos (similitud 1.0) -> Deben agruparse
    # v3 ortogonal (similitud 0.0) -> Debe separarse
    v1 = [1.0, 0.0]
    v2 = [1.0, 0.0]
    v3 = [0.0, 1.0]

    # Configuramos el mock de embed
    # Ollama devuelve {'embeddings': [ [..], [..] ]}
    mock_ollama_client.embed.return_value = {
        'embeddings': [v1, v2, v3]
    }

    # Configuramos el chunker para ser sensible
    chunker.percentile_threshold = 50 # Umbral medio
    
    chunks = chunker.split([doc])

    # Esperamos 2 chunks:
    # 1. "Gato come. Gato duerme." (Similares)
    # 2. "Coche rápido." (Diferente)
    
    assert len(chunks) == 2
    assert "Gato come" in chunks[0].content
    assert "Gato duerme" in chunks[0].content
    assert chunks[1].content == "Coche rápido."

def test_fallback_no_markdown(chunker, mock_ollama_client):
    """Prueba texto plano sin headers."""
    text = "Oración uno. Oración dos."
    doc = Document(content=text)
    
    # Mock embeddings iguales para que no divida
    mock_ollama_client.embed.return_value = {
        'embeddings': [[1,0], [1,0]]
    }
    
    chunks = chunker.split([doc])
    
    assert len(chunks) == 1
    assert chunks[0].content.strip() == text

def test_error_handling_ollama(chunker, mock_ollama_client):
    """Si Ollama falla, debería devolver el chunk entero sin romper."""
    doc = Document(content="Texto prueba.")
    
    # Simular error en la API
    mock_ollama_client.embed.side_effect = Exception("Connection refused")
    
    chunks = chunker.split([doc])
    
    assert len(chunks) == 1
    assert chunks[0].content == "Texto prueba."