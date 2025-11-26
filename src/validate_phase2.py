import asyncio
import sys
import os
from dotenv import load_dotenv

# Aseguramos que Python encuentre el paquete 'srag' si se ejecuta desde src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from srag.core import Chunk
from srag.components.loaders import LlamaParseLoader
from srag.components.embeddings import OllamaEmbeddings
from srag.components.vectorstores import ChromaVectorStore
from srag.components.llms import OllamaLLM

# --- Configuración ---
# Asegúrate de tener estos modelos descargados en Ollama (ollama pull ...)
EMBEDDING_MODEL = "nomic-embed-text" # O "llama3", "mxbai-embed-large", etc.
LLM_MODEL = "llama3.2" # O "llama2", "mistral", "deepseek-r1", etc.
OLLAMA_BASE_URL = "http://172.23.224.1:11434"  # URL del servidor Ollama

async def main():
    print("🚀 Iniciando validación de Fase 2 (Integración RAG)...\n")

    # ---------------------------------------------------------
    # 1. Inicialización de Componentes
    # ---------------------------------------------------------
    print("1️⃣  Inicializando componentes...")

    load_dotenv()  # Carga variables de entorno si es necesario

    # Loader
    try:
        pdf_list = list()  # Simulamos una lista vacía de PDFs para el loader
        pdf_list.append("Hito_1_VC-06.pdf")  # Añadimos un PDF de ejemplo
        loader = LlamaParseLoader(file_paths=pdf_list, save_output=True)
        print("   ✅ Loader LlamaParse listo")
    except Exception as e:
        print(f"   ❌ Error iniciando Loader: {e}")
        return

    # Embedder
    try:
        embedder = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        print(f"   ✅ Embedder listo ({EMBEDDING_MODEL})")
    except Exception as e:
        print(f"   ❌ Error iniciando Embedder: {e}")
        return

    # VectorStore (En memoria para pruebas, sin path ni host)
    try:
        vectorstore = ChromaVectorStore(collection_name="validacion_rapida")
        print("   ✅ ChromaDB (Memoria) listo")
    except Exception as e:
        print(f"   ❌ Error iniciando ChromaDB: {e}")
        return

    # LLM
    try:
        llm = OllamaLLM(model_name=LLM_MODEL, base_url=OLLAMA_BASE_URL)
        print(f"   ✅ LLM listo ({LLM_MODEL})")
    except Exception as e:
        print(f"   ❌ Error iniciando LLM: {e}")
        return

    # ---------------------------------------------------------
    # 2. Creación de Datos de Prueba (Knowledge Base)
    # ---------------------------------------------------------
    print("\n2️⃣  Creando base de conocimiento simulada...")
    
    textos_conocimiento = [
        "La capital de Francia es París y es conocida por la Torre Eiffel.",
        "El Python es un lenguaje de programación interpretado y dinámico.",
        "La fotosíntesis es el proceso por el cual las plantas crean energía del sol.",
        "ChromaDB es una base de datos vectorial open-source diseñada para aplicaciones de IA.",
        "El agua hierve a 100 grados Celsius a nivel del mar."
    ]

    
    try:
        load = await loader.load()  # Método asíncrono
        print(f"   ✅ Loader procesó {len(load)} documentos.")
    except Exception as e:
        print(f"   ❌ Error cargando documentos: {e}")
        return

    chunks = []
    for i, texto in enumerate(textos_conocimiento):
        chunks.append(Chunk(content=texto, metadata={"id_doc": i, "origen": "manual"}))
    
    print(f"   -> Se crearon {len(chunks)} chunks en memoria.")

    # ---------------------------------------------------------
    # 3. Generación de Embeddings (Ingesta)
    # ---------------------------------------------------------
    print("\n3️⃣  Generando Embeddings...")
    
    # Extraemos solo el texto para el embedder
    textos_raw = [c.content for c in chunks]
    
    # Llamada asíncrona a Ollama
    vectores = await embedder.embed_documents(textos_raw)
    
    # Asignamos los vectores calculados a los objetos Chunk
    for chunk, vector in zip(chunks, vectores):
        chunk.embedding = vector
    
    print(f"   ✅ Embeddings generados. Dimensión del vector: {len(vectores[0])}")

    # ---------------------------------------------------------
    # 4. Almacenamiento en VectorStore
    # ---------------------------------------------------------
    print("\n4️⃣  Indexando en ChromaDB...")
    await vectorstore.add(chunks)
    print("   ✅ Datos guardados correctamente.")

    # ---------------------------------------------------------
    # 5. Recuperación (Retrieval)
    # ---------------------------------------------------------
    PREGUNTA = "¿Qué es ChromaDB?"
    print(f"\n5️⃣  Simulando búsqueda para: '{PREGUNTA}'")
    
    # A. Embed de la pregunta
    vector_pregunta = await embedder.embed_query(PREGUNTA)
    
    # B. Búsqueda semántica
    resultados = await vectorstore.search(query_vector=vector_pregunta, k=2)
    
    print("   🔍 Documentos recuperados:")
    contexto_acumulado = ""
    for i, doc in enumerate(resultados):
        print(f"      {i+1}. {doc.content} (ID: {doc.id})")
        contexto_acumulado += f"- {doc.content}\n"

    # ---------------------------------------------------------
    # 6. Generación (RAG)
    # ---------------------------------------------------------
    print("\n6️⃣  Generando respuesta con LLM...")

    prompt_rag = f"""Usa el siguiente contexto para responder a la pregunta de forma concisa.
    
    Contexto:
    {contexto_acumulado}
    
    Pregunta: {PREGUNTA}
    
    Respuesta:"""

    print("-" * 40)
    print("🤖 Respuesta del Modelo (Streaming):")
    
    full_response = ""
    async for token in llm.stream(prompt_rag):
        print(token, end="", flush=True)
        full_response += token
    
    print("\n" + "-" * 40)

    # ---------------------------------------------------------
    # 7. Limpieza (Opcional)
    # ---------------------------------------------------------
    print("\n7️⃣  Limpiando...")
    ids_a_borrar = [c.id for c in chunks]
    await vectorstore.delete(ids_a_borrar)
    print("   ✅ Colección limpiada.")
    print("\n✨ ¡Validación de Fase 2 completada con éxito!")

if __name__ == "__main__":
    asyncio.run(main())