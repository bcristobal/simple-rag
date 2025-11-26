from srag.core import BaseEmbedder
from dotenv import load_dotenv
from ollama import AsyncClient # ✅ Usamos AsyncClient para no bloquear

class OllamaEmbeddings(BaseEmbedder):
    """Implementación de embeddings usando Ollama (Asíncrono)."""

    def __init__(self, model: str = "llama2", base_url: str = None) -> None:
        """
        Inicializa el cliente de Ollama.
        """
        load_dotenv()
        self.model = model
        # Inicializamos el cliente asíncrono
        self.client = AsyncClient(host=base_url)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Genera embeddings para una lista de textos.
        """
        embeddings = []
        for text in texts:
            # 1. CORRECCIÓN: Cambiamos 'text' por 'input'
            # 2. CORRECCIÓN: Añadimos 'await'
            response = await self.client.embed(model=self.model, input=text)
            
            # 3. CORRECCIÓN: La respuesta de embed es un dict con una lista de listas
            # Accedemos a 'embeddings' y tomamos el primer elemento (el único que pedimos)
            embeddings.append(response['embeddings'][0])
            
        return embeddings

    async def embed_query(self, text: str) -> list[float]:
        """
        Genera un embedding para una única consulta.
        """
        # 1. CORRECCIÓN: 'input' en lugar de 'text' y uso de 'await'
        response = await self.client.embed(model=self.model, input=text)
        
        # 2. Retornamos el primer vector
        return response['embeddings'][0]