
# 🧠 SRAG (Simple RAG Library)

**SRAG** es una librería de Python modular, asíncrona y fuertemente tipada diseñada para construir aplicaciones de **Retrieval Augmented Generation (RAG)** de manera escalable y mantenible.

A diferencia de los scripts monolíticos, SRAG desacopla la lógica en componentes intercambiables (Loaders, Chunkers, Embedders, VectorStores, LLMs), permitiendo crear pipelines complejos con facilidad.

## ✨ Características Principales

  * **⚡ 100% Asíncrono:** Construido sobre `asyncio` para operaciones I/O no bloqueantes (ideal para APIs y alta concurrencia).
  * **🧩 Diseño Modular:** Basado en interfaces abstractas (`ABC`). Cambia de `Ollama` a `OpenAI` o de `ChromaDB` a `Pinecone` sin romper tu lógica de negocio.
  * **🛡️ Type-Safe:** Uso extensivo de **Pydantic** para validación de datos y **Type Hints** para una experiencia de desarrollo robusta.
  * **📄 LlamaParse Integration:** Soporte nativo para parsing avanzado de documentos (PDF, tablas) a Markdown.

## 📦 Estructura del Proyecto

```text
src/srag/
├── core/           # Contratos e Interfaces (BaseLLM, BaseLoader...)
├── components/     # Implementaciones Concretas
│   ├── loaders/    # LlamaParseLoader, etc.
│   ├── chunkers/   # FixedLengthChunker, etc.
│   ├── embeddings/ # OllamaEmbeddings, etc.
│   ├── vectorstores/ # ChromaVectorStore, etc.
│   └── llms/       # OllamaLLM, etc.
└── pipeline/       # (Próximamente) Orquestación de Ingesta
```

## 🚀 Inicio Rápido

### Prerrequisitos

1.  **Python 3.10+**
2.  **Ollama** ejecutándose localmente (para Embeddings y LLM).
3.  **LlamaCloud API Key** (si usas el loader de PDFs).

### Instalación

Si usas `uv` (recomendado) o `pip`:

```bash
# Instalar dependencias
pip install ollama chromadb llama-parse pydantic python-dotenv pymupdf
```

Crea un archivo `.env` en la raíz:

```env
LLAMA_CLOUD_API_KEY=llx-tu-api-key-aqui
```

### Ejemplo de Uso (End-to-End)

Este ejemplo muestra cómo cargar un PDF, dividirlo, vectorizarlo y chatear con él.

```python
import asyncio
from srag.components.loaders import LlamaParseLoader
from srag.components.chunkers import FixedLengthChunker
from srag.components.embeddings import OllamaEmbeddings
from srag.components.vectorstores import ChromaVectorStore
from srag.components.llms import OllamaLLM

async def main():
    # 1. Configuración de Componentes
    loader = LlamaParseLoader(file_paths=["documento.pdf"], save_output=False)
    chunker = FixedLengthChunker(chunk_size=500, overlap=50)
    embedder = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = ChromaVectorStore(collection_name="demo_rag")
    llm = OllamaLLM(model_name="llama3.2")

    # 2. Ingesta (Load -> Split -> Embed -> Store)
    print("📥 Cargando y procesando...")
    docs = await loader.load()
    chunks = chunker.split(docs)
    
    vectors = await embedder.embed_documents([c.content for c in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk.embedding = vector
        
    await vectorstore.add(chunks)
    print(f"✅ Indexados {len(chunks)} fragmentos.")

    # 3. Chat (Retrieve -> Generate)
    query = "¿Cuáles son los puntos clave del documento?"
    print(f"\nPregunta: {query}")
    
    # Retrieval
    query_vec = await embedder.embed_query(query)
    results = await vectorstore.search(query_vec, k=3)
    context = "\n".join([c.content for c in results])
    
    # Generation (Streaming)
    prompt = f"Contexto: {context}\n\nPregunta: {query}\nRespuesta:"
    print("Respuesta: ", end="", flush=True)
    async for token in llm.stream(prompt):
        print(token, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
```

## 🛠️ Desarrollo y Testing

El proyecto utiliza `pytest` para las pruebas unitarias y de integración.

```bash
# Ejecutar todos los tests
uv run pytest

# Ejecutar tests de un componente específico
uv run pytest tests/components/loaders/
```

## 🗺️ Roadmap

  * [x]  Definición de interfaces core (`core`) y tipos de datos (`types`).
  * [x]  Implementación de Componentes Base (Ollama, Chroma, LlamaParse).
  * [x]  Tests Unitarios asíncronos.
  * [ ]  **Pipeline de Ingesta:** Orquestador automático ETL.
  * [ ]  **Estrategias RAG:** Implementación de patrones como *Simple RAG*, *Hybrid Search* y *Contextual RAG*.
