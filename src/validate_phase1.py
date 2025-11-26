from src.srag.core.types import Document, Chunk
from src.srag.core.interfaces import BaseLoader
from typing import List

# 1. Probando Pydantic
try:
    doc = Document(content="Hola mundo", metadata={"source": "test"})
    print("✅ Documento creado con éxito:", doc)
    
    chunk = Chunk(content="Fragmento", document_id=doc.id, index=1)
    print("✅ Chunk creado con éxito:", chunk)
    
    # Prueba de validación (esto debería fallar si content no es str)
    # bad_doc = Document(content=123) 
except Exception as e:
    print(f"❌ Error en Pydantic: {e}")

# 2. Probando Interfaces (Mock simple)
class MockLoader(BaseLoader):
    def load(self) -> List[Document]:
        return [Document(content="Mock Data")]

try:
    loader = MockLoader()
    docs = loader.load()
    print(f"✅ Interfaz BaseLoader implementada correctamente. Datos: {docs[0].content}")
except TypeError as e:
    print(f"❌ Error de implementación de interfaz: {e}")

print("\n🎉 FASE 1 COMPLETADA: Estructura, Tipos e Interfaces listas.")