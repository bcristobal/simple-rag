from abc import ABC, abstractmethod
from typing import List, AsyncGenerator, Optional
from .types import Document, Chunk, Vector, Filters

class BaseLoader(ABC):
    """
    Interfaz para cargar documentos.
    La configuración de la fuente (URLs, rutas de archivos, credenciales) 
    debe ocurrir en el __init__ de la clase concreta.
    """
    
    @abstractmethod
    async def load(self, url_file: List[str]) -> List[Document]:
        """
        Carga datos desde la fuente configurada y retorna documentos.
        Debe ser asíncrono para permitir I/O no bloqueante (ej. web scraping).
        """
        pass

class BaseChunker(ABC):
    """
    Interfaz para dividir documentos en chunks.
    Generalmente es una tarea intensiva en CPU, por lo que puede ser síncrona,
    pero la definimos flexible.
    """
    
    @abstractmethod
    async def split(self, documents: List[Document]) -> List[Chunk]:
        """Recibe documentos completos y retorna fragmentos."""
        pass

class BaseEmbedder(ABC):
    """Interfaz para convertir texto a vectores."""
    
    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[Vector]:
        """
        Genera embeddings para una lista de textos crudos.
        Se recibe List[str] en lugar de List[Chunk] para desacoplar la lógica.
        """
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> Vector:
        """Genera embedding para una única consulta (búsqueda)."""
        pass

class BaseVectorStore(ABC):
    """Interfaz para almacenamiento y recuperación vectorial."""
    
    @abstractmethod
    async def add(self, chunks: List[Chunk]) -> None:
        """
        Almacena chunks en la base de datos. 
        Asume que los chunks ya tienen el campo .embedding calculado.
        """
        pass

    @abstractmethod
    async def search(
        self, 
        query_vector: Vector, 
        k: int = 4,
        filters: Optional[Filters] = None
    ) -> List[Chunk]:
        """
        Retorna los K chunks más similares.
        
        Args:
            query_vector: El vector de la pregunta del usuario.
            k: Número de resultados.
            filters: Diccionario opcional para filtrar por metadatos (ej. {'author': 'admin'}).
        """
        pass
        
    @abstractmethod
    async def delete(self, doc_ids: List[str]) -> bool:
        """Elimina chunks basados en los IDs de los chunks o documentos padre."""
        pass

class BaseLLM(ABC):
    """Interfaz para Modelos de Lenguaje (Generación)."""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Genera una respuesta completa (espera hasta terminar)."""
        pass

    @abstractmethod
    async def stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """
        Genera respuesta token a token.
        Retorna un generador asíncrono para interfaces tipo chat.
        """
        pass