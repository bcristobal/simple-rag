from typing import AsyncGenerator, Any, Optional
import os

# Importaciones de LangChain y Google GenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Importamos la interfaz base (asumiendo que srag.core es la ruta correcta según tu archivo ollama_llms.py)
from srag.core import BaseLLM

class GeminiLLM(BaseLLM):
    """
    Implementación de BaseLLM usando Google Gemini a través de LangChain.
    Requiere la librería: langchain-google-genai
    """

    def __init__(
        self, 
        model_name: str = "gemini-2.5-flash", 
        api_key: Optional[str] = None, 
        temperature: float = 0.3,
        **kwargs: Any
    ) -> None:
        """
        Inicializa el modelo de Gemini.
        
        Args:
            model_name: Nombre del modelo (ej. "gemini-1.5-flash", "gemini-pro").
            api_key: API Key de Google. Si es None, busca GOOGLE_API_KEY en variables de entorno.
            temperature: Creatividad del modelo.
            kwargs: Argumentos adicionales soportados por ChatGoogleGenerativeAI.
        """
        # Si no se pasa api_key, intentamos obtenerla del entorno, útil si usas load_dotenv() fuera.
        google_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        self.model_name = model_name
        
        # Inicializamos el cliente de LangChain
        self.client = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=google_api_key,
            temperature=temperature,
            **kwargs
        )

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Genera una respuesta completa utilizando ainvoke de LangChain.
        """
        # Convertimos el string prompt a un mensaje de usuario de LangChain
        messages = [HumanMessage(content=prompt)]
        
        # ainvoke es el método asíncrono estándar de LangChain
        response = await self.client.ainvoke(messages, **kwargs)
        
        # response.content contiene el texto de la respuesta (str)
        return response.content
    
    async def stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """
        Genera texto de manera secuencial utilizando astream de LangChain.
        """
        messages = [HumanMessage(content=prompt)]
        
        # astream retorna un generador asíncrono de chunks
        async for chunk in self.client.astream(messages, **kwargs):
            # En LangChain, chunk.content es el fragmento de texto
            if chunk.content:
                yield chunk.content
                