import uuid
from typing import List, Optional, Dict, Any, TypeAlias
from pydantic import BaseModel, Field, ConfigDict

# Alias de tipos para mejorar la legibilidad en las interfaces
Metadata: TypeAlias = Dict[str, Any]
Vector: TypeAlias = List[float]
Filters: TypeAlias = Dict[str, Any]

class RAGBaseModel(BaseModel):
    """Configuración base para todos los modelos internos."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True
    )

class Document(RAGBaseModel):
    """
    Representa un documento original cargado.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: Metadata = Field(default_factory=dict)
    
    def __bool__(self):
        return bool(self.content)

class Chunk(RAGBaseModel):
    """
    Representa un fragmento de texto procesado.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: Optional[str] = None  # Referencia al documento padre
    content: str
    metadata: Metadata = Field(default_factory=dict)
    
    # El vector es estrictamente una lista de floats (o None si no se ha calculado)
    embedding: Optional[Vector] = None 
    
    index: int = 0  # Orden del chunk dentro del documento original

    def __repr__(self):
        emb_status = "Set" if self.embedding else "None"
        return f"Chunk(id={self.id[:8]}, content='{self.content[:30]}...', embedding={emb_status})"