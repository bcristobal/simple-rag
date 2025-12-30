import os
from typing import Any, AsyncGenerator, Optional
from dotenv import load_dotenv
from huggingface_hub import AsyncInferenceClient
from srag.core import BaseLLM

class HuggingFaceLLM(BaseLLM):
    """
    Implementación de BaseLLM usando la API de Inferencia de Hugging Face.
    Ideal para modelos como 'mistralai/Mistral-7B-Instruct-v0.2' o 'meta-llama/Meta-Llama-3-8B-Instruct'.
    """

    def __init__(
        self, 
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3", 
        api_key: Optional[str] = None
    ) -> None:
        load_dotenv()
        token = api_key or os.getenv("HF_TOKEN")
        self.model_name = model_name
        self.client = AsyncInferenceClient(token=token)

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Genera una respuesta completa usando text_generation.
        """
        # Mapeamos kwargs comunes a los parámetros de HF si es necesario
        # max_new_tokens es el estándar en HF
        if "max_tokens" in kwargs:
            kwargs["max_new_tokens"] = kwargs.pop("max_tokens")
            
        response = await self.client.text_generation(
            prompt=prompt,
            model=self.model_name,
            stream=False,
            **kwargs
        )
        return response

    async def stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """
        Genera texto de manera secuencial (streaming).
        """
        if "max_tokens" in kwargs:
            kwargs["max_new_tokens"] = kwargs.pop("max_tokens")

        stream = await self.client.text_generation(
            prompt=prompt,
            model=self.model_name,
            stream=True,
            details=True, # Para obtener detalles del token si fuera necesario
            **kwargs
        )
        
        async for response in stream:
            # response.token.text contiene el fragmento de texto generado
            content = response.token.text
            if content:
                yield content