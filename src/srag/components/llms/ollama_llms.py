from typing import AsyncGenerator, Any
from ollama import AsyncClient
from srag.core import BaseLLM

class OllamaLLM(BaseLLM):
    """Implementación de BaseLLM usando Ollama (Asíncrono)."""

    def __init__(self, model_name: str, base_url: str = None) -> None:
        self.model_name = model_name
        # IMPORTANTE: Usamos el cliente asíncrono
        self.client = AsyncClient(host=base_url)

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Genera una respuesta completa."""
        response = await self.client.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            stream=False,
            options=kwargs
        )
        return response['message']['content']
    
    async def stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """Genera texto de manera secuencial (streaming)."""
        stream = await self.client.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True,
            options=kwargs
        )
        
        async for chunk in stream:
            # Accedemos al diccionario de respuesta de forma segura
            msg = chunk.get('message', {})
            content = msg.get('content', '')
            
            # Si hay contenido, lo enviamos al usuario
            if content:
                yield content