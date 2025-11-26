import pytest
from unittest.mock import patch, MagicMock
from srag.components.llms.ollama_llms import OllamaLLM

# Fixture para simular el AsyncClient de Ollama
@pytest.fixture
def mock_ollama_client():
    # 1. Patcheamos la clase AsyncClient
    with patch('srag.components.llms.ollama_llms.AsyncClient') as MockClient:
        # 2. Obtenemos la instancia simulada que se crea dentro de __init__
        mock_instance = MockClient.return_value
        yield mock_instance

@pytest.mark.asyncio
async def test_stream_generation(mock_ollama_client):
    """Prueba que el generador hace yield de los tokens correctamente."""
    
    # Datos simulados
    mock_chunks = [
        {'message': {'content': 'Hola'}},
        {'message': {'content': ' mun'}},
        {'message': {'content': 'do'}},
        {'done': True}
    ]

    # --- EL TRUCO ESTÁ AQUÍ ---
    # 1. Creamos el generador asíncrono real (el flujo de datos)
    async def real_async_generator():
        for chunk in mock_chunks:
            yield chunk

    # 2. Creamos una función asíncrona que DEVUELVE ese generador.
    # Esto simula el comportamiento de: await client.chat(...) -> retorna stream
    async def mock_chat_call(*args, **kwargs):
        return real_async_generator()

    # 3. Asignamos esta función al side_effect del mock
    mock_ollama_client.chat.side_effect = mock_chat_call
    # --------------------------

    # Instanciamos
    llm = OllamaLLM(model_name="dummy-model")

    # Consumimos el stream
    collected_text = ""
    async for token in llm.stream("Dime hola"):
        collected_text += token

    # Verificaciones
    assert collected_text == "Hola mundo"
    
    mock_ollama_client.chat.assert_called_once_with(
        model="dummy-model",
        messages=[{'role': 'user', 'content': 'Dime hola'}],
        stream=True,
        options={}
    )

@pytest.mark.asyncio
async def test_generate_simple(mock_ollama_client):
    """Prueba la generación normal (sin stream)."""
    
    # --- EL TRUCO ESTÁ AQUÍ ---
    # Definimos una función asíncrona que devuelve el dict final.
    # Esto satisface el: response = await client.chat(...)
    async def mock_chat_return(*args, **kwargs):
        return {'message': {'content': 'Respuesta completa'}}

    mock_ollama_client.chat.side_effect = mock_chat_return
    # --------------------------

    llm = OllamaLLM(model_name="dummy-model")
    response = await llm.generate("Test")

    assert response == "Respuesta completa"