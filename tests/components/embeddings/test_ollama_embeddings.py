import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from srag.core import BaseEmbedder
from srag.components.embeddings.ollama_embeddings import OllamaEmbeddings

# --- Fixtures ---

@pytest.fixture
def mock_ollama_setup():
    """
    Mockea la clase AsyncClient de la librería ollama.
    """
    # 1. CAMBIO: Apuntamos a AsyncClient
    patch_path = 'srag.components.embeddings.ollama_embeddings.AsyncClient'
    
    with patch(patch_path) as MockClientClass:
        mock_client_instance = MockClientClass.return_value
        
        # 2. CAMBIO: Configuramos 'embed' como AsyncMock para permitir 'await'
        mock_client_instance.embed = AsyncMock()
        
        yield MockClientClass, mock_client_instance

@pytest.fixture
def embedder(mock_ollama_setup):
    """Retorna una instancia con el cliente mockeado."""
    return OllamaEmbeddings(model="test-model", base_url="http://localhost:11434")

# --- Tests ---

def test_init(mock_ollama_setup):
    """Prueba que el cliente AsyncClient se inicializa correctamente."""
    MockClientClass, _ = mock_ollama_setup
    
    # Instanciamos
    OllamaEmbeddings(model="llama3", base_url="http://mi-servidor:11434")
    
    # Verificamos llamada al constructor
    MockClientClass.assert_called_with(host="http://mi-servidor:11434")

def test_inheritance():
    """Verifica la herencia de la interfaz base."""
    # CAMBIO: Parcheamos AsyncClient para instanciar sin error
    with patch('srag.components.embeddings.ollama_embeddings.AsyncClient'):
        emb = OllamaEmbeddings()
        assert isinstance(emb, BaseEmbedder)

@pytest.mark.asyncio
async def test_embed_query(embedder, mock_ollama_setup):
    """Prueba embed_query (un solo texto)."""
    _, mock_client_instance = mock_ollama_setup
    
    # 3. CAMBIO: La respuesta ahora es un diccionario con lista de listas
    # (Simulando la respuesta real de la API de Ollama)
    mock_response = {'embeddings': [[0.1, 0.2, 0.3]]}
    
    mock_client_instance.embed.return_value = mock_response

    # Ejecutamos
    result = await embedder.embed_query("Hola mundo")

    # Verificamos
    assert result == [0.1, 0.2, 0.3]
    
    # 4. CAMBIO: Verificamos que se usó 'input' en lugar de 'text'
    mock_client_instance.embed.assert_called_once_with(
        model="test-model", 
        input="Hola mundo"
    )

@pytest.mark.asyncio
async def test_embed_documents(embedder, mock_ollama_setup):
    """Prueba embed_documents (lista de textos)."""
    _, mock_client_instance = mock_ollama_setup
    
    vec1 = [1.0, 1.0]
    vec2 = [2.0, 2.0]
    
    # 3. CAMBIO: Respuestas simuladas como diccionarios
    resp1 = {'embeddings': [vec1]}
    resp2 = {'embeddings': [vec2]}
    
    # Configuramos side_effect
    mock_client_instance.embed.side_effect = [resp1, resp2]

    # Ejecutamos
    texts = ["Texto A", "Texto B"]
    results = await embedder.embed_documents(texts)

    # Verificamos
    assert len(results) == 2
    assert results[0] == vec1
    assert results[1] == vec2
    
    # 4. CAMBIO: Validamos las llamadas con 'input'
    assert mock_client_instance.embed.call_count == 2
    
    # Opcional: verificar argumentos exactos
    from unittest.mock import call
    expected_calls = [
        call(model="test-model", input="Texto A"),
        call(model="test-model", input="Texto B")
    ]
    mock_client_instance.embed.assert_has_calls(expected_calls)