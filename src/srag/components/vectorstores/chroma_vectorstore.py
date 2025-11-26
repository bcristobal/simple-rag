from srag.core import BaseVectorStore
from srag.core import Chunk
import chromadb

class ChromaVectorStore(BaseVectorStore):
    """Implementación de BaseVectorStore usando ChromaDB."""

    def __init__(self, collection_name: str, path: str = None, host: str = None, port: int = None) -> None:
        """
        Inicializa el ChromaDB client y la colección.
        
        Args:
            client (Client): Instancia del cliente ChromaDB.
            collection_name (str): Nombre de la colección a usar.
            path (str, optional): Ruta para almacenar la base de datos de ChromaDB. Defaults to None.
            host (str, optional): Host para conexión HTTP. Defaults to None.
            port (int, optional): Puerto para conexión HTTP. Defaults to None.
        """
        if host and port:
            self.client = chromadb.HttpClient(host=host, port=port)
        elif path:
            self.client = chromadb.PersistentClient(path=path)
        else:
            self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)

    async def add(self, chunks: list) -> None:
        """Agrega chunks a la colección de ChromaDB."""
        ids = [chunk.id for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        documents = [chunk.content for chunk in chunks]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

    async def search(self, query_vector: list[float], k: int = 4, filters: dict = None) -> list:
        """Busca los K chunks más similares en la colección."""
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where=filters
        )
        found_chunks = []
        for doc_id, doc_text, metadata in zip(results['ids'][0], results['documents'][0], results['metadatas'][0]):
            chunk = Chunk(id=doc_id, content=doc_text, metadata=metadata, embedding=None)
            found_chunks.append(chunk)
        return found_chunks

    async def delete(self, doc_ids: list[str]) -> bool:
        """Elimina chunks basados en los IDs proporcionados."""
        self.collection.delete(ids=doc_ids)
        return True