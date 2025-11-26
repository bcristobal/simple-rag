import pytest
from unittest.mock import patch, MagicMock, call
from srag.core import BaseVectorStore
# Ajusta el import a tu ruta real
from srag.components.vectorstores.chroma_vectorstore import ChromaVectorStore

# --- Helper para Chunks ---
class MockChunk:
    """Simula un objeto Chunk simple."""
    def __init__(self, id, content, metadata, embedding):
        self.id = id
        self.content = content
        self.metadata = metadata
        self.embedding = embedding

# --- 1. Tests de Lógica de Inicialización (__init__) ---

def test_init_http_client():
    """Prueba que se cree un HttpClient si se pasan host y port."""
    with patch('srag.components.vectorstores.chroma_vectorstore.chromadb') as mock_chroma:
        
        # Ejecutar
        store = ChromaVectorStore(collection_name="test", host="localhost", port=8000)
        
        # Verificar que usó HttpClient
        mock_chroma.HttpClient.assert_called_once_with(host="localhost", port=8000)
        # Verificar que NO usó los otros
        mock_chroma.PersistentClient.assert_not_called()
        mock_chroma.Client.assert_not_called()
        
        # Verificar que creó la colección
        store.client.get_or_create_collection.assert_called_once_with(name="test")

def test_init_persistent_client():
    """Prueba que se cree un PersistentClient si se pasa path."""
    with patch('srag.components.vectorstores.chroma_vectorstore.chromadb') as mock_chroma:
        
        # Ejecutar
        store = ChromaVectorStore(collection_name="test", path="./my_db")
        
        # Verificar que usó PersistentClient
        mock_chroma.PersistentClient.assert_called_once_with(path="./my_db")
        mock_chroma.HttpClient.assert_not_called()

def test_init_ephemeral_client():
    """Prueba que se cree un Client (en memoria) si no se pasan argumentos."""
    with patch('srag.components.vectorstores.chroma_vectorstore.chromadb') as mock_chroma:
        
        # Ejecutar
        store = ChromaVectorStore(collection_name="test")
        
        # Verificar que usó Client básico
        mock_chroma.Client.assert_called_once()
        mock_chroma.PersistentClient.assert_not_called()

# --- 2. Tests de Funcionalidad (Add, Search, Delete) ---

@pytest.fixture
def vectorstore_setup():
    """
    Fixture que prepara un VectorStore con un cliente mockeado
    listo para probar los métodos async.
    """
    with patch('srag.components.vectorstores.chroma_vectorstore.chromadb') as mock_chroma:
        # Simulamos el cliente "InMemory" por defecto para estos tests
        mock_client_instance = mock_chroma.Client.return_value
        mock_collection = mock_client_instance.get_or_create_collection.return_value
        
        store = ChromaVectorStore(collection_name="func-test")
        
        yield store, mock_collection

@pytest.mark.asyncio
async def test_add(vectorstore_setup):
    store, mock_collection = vectorstore_setup

    chunks = [
        MockChunk(id="1", content="Texto A", metadata={"meta": "a"}, embedding=[0.1, 0.2]),
        MockChunk(id="2", content="Texto B", metadata={"meta": "b"}, embedding=[0.3, 0.4]),
    ]

    await store.add(chunks)

    mock_collection.add.assert_called_once_with(
        ids=["1", "2"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        metadatas=[{"meta": "a"}, {"meta": "b"}],
        documents=["Texto A", "Texto B"]
    )

@pytest.mark.asyncio
async def test_search(vectorstore_setup):
    store, mock_collection = vectorstore_setup

    # Simulamos la respuesta compleja de Chroma (Listas de listas)
    mock_collection.query.return_value = {
        'ids': [['id1', 'id2']],
        'documents': [['Contenido 1', 'Contenido 2']],
        'metadatas': [[{'page': 10}, {'page': 11}]]
    }

    query_vec = [0.9, 0.9]
    results = await store.search(query_vector=query_vec, k=2)

    # Verificar llamada a Chroma
    mock_collection.query.assert_called_once_with(
        query_embeddings=[query_vec],
        n_results=2,
        where=None
    )

    # Verificar reconstrucción de objetos Chunk
    assert len(results) == 2
    
    assert results[0].id == "id1"
    assert results[0].content == "Contenido 1"
    assert results[0].metadata == {'page': 10}
    
    assert results[1].id == "id2"
    assert results[1].content == "Contenido 2"

@pytest.mark.asyncio
async def test_delete(vectorstore_setup):
    store, mock_collection = vectorstore_setup
    
    ids_to_del = ["a", "b"]
    await store.delete(ids_to_del)
    
    mock_collection.delete.assert_called_once_with(ids=ids_to_del)