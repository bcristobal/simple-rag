import os
from typing import List, Optional
from dotenv import load_dotenv
from huggingface_hub import AsyncInferenceClient
from srag.core import BaseEmbedder

class HuggingFaceEmbeddings(BaseEmbedder):
    """
    Implementación de BaseEmbedder usando la API de Inferencia de Hugging Face.
    """

    def __init__(
        self, 
        model: str = "sentence-transformers/all-MiniLM-L6-v2", 
        api_key: Optional[str] = None
    ) -> None:
        """
        Args:
            model: ID del modelo en Hugging Face Hub.
            api_key: Token de HF (HF_TOKEN). Si es None, busca en variables de entorno.
        """
        load_dotenv()
        token = api_key or os.getenv("HF_TOKEN")
        self.model = model
        self.client = AsyncInferenceClient(token=token)

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para una lista de textos.
        """
        # La API de feature_extraction puede procesar listas, 
        # pero es mejor manejarlo con cuidado para no exceder límites de la API.
        embeddings = []
        
        # Nota: Dependiendo del modelo en HF, feature_extraction puede devolver 
        # (batch, seq_len, dim) o (batch, dim). Los modelos de sentence-transformers
        # optimizados para la API suelen devolver ya el pooled embedding.
        result = await self.client.feature_extraction(texts, model=self.model)
        
        # Si el resultado es un numpy array o tensor, lo convertimos a lista
        if hasattr(result, "tolist"):
            result = result.tolist()
            
        # Nos aseguramos de que sea una lista de listas de floats
        # En algunos casos la API devuelve [batch, tokens, dim], si es así,
        # habría que hacer un mean pooling. Asumimos aquí que el modelo devuelve
        # vectores de sentencia (común en modelos de embedding modernos en la API).
        return result

    async def embed_query(self, text: str) -> List[float]:
        """
        Genera un embedding para una única consulta.
        """
        result = await self.client.feature_extraction(text, model=self.model)
        
        if hasattr(result, "tolist"):
            result = result.tolist()
            
        # Si la API devuelve una lista de listas (aunque sea un solo texto),
        # tomamos el primer elemento.
        if isinstance(result[0], list):
            return result[0]
            
        return result