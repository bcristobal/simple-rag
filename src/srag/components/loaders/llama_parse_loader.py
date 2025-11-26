import os
import pymupdf  # Requiere: pip install pymupdf
from typing import List
from llama_parse import LlamaParse # Requiere: pip install llama-parse
from srag.core import BaseLoader, Document

class LlamaParseLoader(BaseLoader):
    """
    Carga documentos PDF/Docs usando LlamaParse.
    Genera IDs estables basados en contenido y permite guardar backup en Markdown.
    """

    def __init__(self, file_paths: List[str], save_output: bool = False, **kwargs):
        """
        Args:
            file_paths: Lista de rutas de archivos a procesar.
            save_output: Si True, guarda un .md con el resultado junto al archivo original.
            **kwargs: Argumentos extra para LlamaParse (ej. api_key, verbose).
        """
        self.file_paths = file_paths
        self.save_output = save_output
        
        # Configuración del parser
        instructions = """
        Intenta extraer la jerarquía de encabezados correctamente usando formato Markdown.
        Mantén las tablas y listas con su formato Markdown correspondiente.
        """
        
        self.parser = LlamaParse(
            result_type="markdown",
            system_prompt=instructions,
            verbose=kwargs.get("verbose", True),
        )

    async def load(self) -> List[Document]:
        srag_documents = []

        for file_path in self.file_paths:
            if not os.path.exists(file_path):
                print(f"⚠️  Archivo no encontrado: {file_path}")
                continue

            # 1. Parsear (Llamada a la API)
            try:
                # aload_data devuelve una lista de objetos LlamaIndex Document
                llama_docs = await self.parser.aload_data(file_path)
            except Exception as e:
                print(f"❌ Error parseando {file_path}: {e}")
                continue

            if not llama_docs:
                continue

            # 2. Unificar texto (LlamaParse a veces devuelve 1 obj por página)
            full_text = "\n\n".join([doc.text for doc in llama_docs])

            # 3. Guardar backup opcional
            if self.save_output:
                self._save_markdown(file_path, full_text)

            # 4. Generar ID estable
            doc_id = self._get_doc_id(file_path)

            # 5. Crear Documento SRAG
            # Usamos metadatos del primer fragmento y añadimos los nuestros
            metadata = llama_docs[0].metadata.copy()
            metadata.update({
                "source": file_path,
                "loader": "LlamaParse"
            })

            srag_documents.append(
                Document(
                    id=doc_id,
                    content=full_text,
                    metadata=metadata,
                )
            )

        return srag_documents

    def _get_doc_id(self, file_path: str) -> str:
        """Genera hash SHA256 del contenido del PDF (requiere pymupdf)."""
        import hashlib
        try:
            with pymupdf.open(file_path) as doc:
                text = "".join(page.get_text() for page in doc)
                return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        except Exception:
            # Fallback si no es PDF o falla pymupdf: hash del path + tamaño
            stat = os.stat(file_path)
            raw = f"{file_path}_{stat.st_size}"
            return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _save_markdown(self, file_path: str, text: str):
        """Guarda el contenido extraído en un archivo .md"""
        new_path = file_path + ".md"
        try:
            with open(new_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            print(f"⚠️ No se pudo guardar backup Markdown: {e}")