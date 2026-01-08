from typing import List
from srag.core import BaseChunker, Document, Chunk
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import Document as LlamaDocument # Clase documento de LlamaIndex

class LlamaIndexGraphChunker(BaseChunker):
    """
    Usa la inteligencia de LlamaIndex para parsear la jerarquía Markdown.
    Convierte el texto en Nodos y preserva la relación Padre-Hijo en los metadatos.
    """
    
    def __init__(self):
        self.parser = MarkdownNodeParser()

    async def split(self, documents: List[Document]) -> List[Chunk]:
        final_chunks = []
        
        for doc in documents:
            # 1. Convertir nuestro Documento SRAG a Documento LlamaIndex
            llama_doc = LlamaDocument(text=doc.content, metadata=doc.metadata)
            
            # 2. Obtener Nodos (LlamaIndex parsea el árbol MD aquí)
            nodes = self.parser.get_nodes_from_documents([llama_doc])
            
            # 3. Convertir Nodos de vuelta a Chunks SRAG
            for node in nodes:
                # El MarkdownNodeParser extrae metadatos increíbles automáticamente.
                # Ejemplo de node.metadata: 
                # {'Header_1': 'TÍTULO PRELIMINAR', 'Header_2': 'Artículo 4'}
                
                # Construimos un "breadcrumbs" (migas de pan) para el contexto
                headers = []
                for key in sorted(node.metadata.keys()):
                    if "Header" in key:
                        headers.append(node.metadata[key])
                
                context_str = " > ".join(headers)
                
                # Inyectamos el contexto explícitamente en el texto para el embedding
                if context_str:
                    content_with_context = f"CONTEXTO: {context_str}\n---\n{node.text}"
                else:
                    content_with_context = node.text

                chunk = Chunk(
                    content=content_with_context,
                    document_id=doc.id,
                    metadata=node.metadata, # Guardamos la jerarquía estructurada
                    # index se puede calcular después
                )
                final_chunks.append(chunk)
                
        return final_chunks