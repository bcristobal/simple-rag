import asyncio
import sys
import os
from dotenv import load_dotenv

# Aseguramos que Python encuentre el paquete 'srag'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importamos TODOS los componentes necesarios
from srag.components.loaders import LlamaParseLoader
from srag.components.chunkers.fixed_length_chunker import FixedLengthChunker
from srag.components.embeddings import OllamaEmbeddings
from srag.components.vectorstores import ChromaVectorStore
from srag.components.llms import OllamaLLM

# --- CONFIGURACIÓN ---
PDF_FILE = "Hito_1_VC-06.pdf"  # Asegúrate de que este archivo existe en la raíz o ruta correcta
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://172.23.224.1:11434"
COLLECTION_NAME = "fase3_pipeline_completo"

async def run_pipeline():
    print("🚀 Iniciando Validación Fase 3: Pipeline Completo (End-to-End)\n")
    load_dotenv()

    # ==========================================
    # 1. INICIALIZACIÓN DE COMPONENTES
    # ==========================================
    print("1️⃣  Inicializando componentes...")
    
    try:
        # A. Loader (Documentos)
        if not os.path.exists(PDF_FILE):
            print(f"❌ Error: No se encuentra el archivo '{PDF_FILE}'. Por favor añádelo.")
            return
        
        loader = LlamaParseLoader(file_paths=[PDF_FILE], save_output=True)
        
        # B. Chunker (División)
        chunker = FixedLengthChunker(chunk_size=500, overlap=50)
        
        # C. Embedder (Vectores)
        embedder = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        
        # D. VectorStore (Base de Datos)
        vectorstore = ChromaVectorStore(collection_name=COLLECTION_NAME)
        
        # E. LLM (Cerebro)
        llm = OllamaLLM(model_name=LLM_MODEL, base_url=OLLAMA_BASE_URL)
        
        print("   ✅ Todos los componentes iniciados correctamente.")

    except Exception as e:
        print(f"   ❌ Error en inicialización: {e}")
        return

    # ==========================================
    # 2. INGESTA DE DATOS (ETL)
    # ==========================================
    print("\n2️⃣  Ejecutando Pipeline de Ingesta...")

    # Paso A: Cargar
    print(f"   📥 Cargando {PDF_FILE} con LlamaParse...")
    raw_docs = await loader.load()
    if not raw_docs:
        print("   ❌ No se cargaron documentos.")
        return
    print(f"      -> Documentos crudos: {len(raw_docs)}")

    # Paso B: Chunking
    print("   ✂️  Dividiendo en chunks...")
    chunks = chunker.split(raw_docs)
    print(f"      -> Chunks generados: {len(chunks)}")
    print(f"      -> Ejemplo chunk 1: {chunks[0].content[:100]}...")

    # Paso C: Embedding
    print(f"   🧠 Generando embeddings con {EMBEDDING_MODEL}...")
    textos = [c.content for c in chunks]
    vectores = await embedder.embed_documents(textos)
    
    # Asignar vectores a los chunks
    for chunk, vector in zip(chunks, vectores):
        chunk.embedding = vector
    print("      -> Embeddings calculados y asignados.")

    # Paso D: Indexado
    print(f"   💾 Guardando en ChromaDB ({COLLECTION_NAME})...")
    await vectorstore.add(chunks)
    print("   ✅ Ingesta completada.")

    # ==========================================
    # 3. INTERACCIÓN (RAG)
    # ==========================================
    print("\n3️⃣  Probando el Chat RAG...")
    
    # Pregunta de prueba (Idealmente relacionada con tu PDF)
    PREGUNTA = "¿De qué trata el documento y cuáles son los puntos clave?"
    print(f"   ❓ Pregunta: '{PREGUNTA}'")

    # Paso A: Retrieve
    print("   🔍 Buscando contexto relevante...")
    query_vec = await embedder.embed_query(PREGUNTA)
    context_chunks = await vectorstore.search(query_vec, k=3)
    
    context_text = "\n---\n".join([c.content for c in context_chunks])
    print(f"      -> Encontrados {len(context_chunks)} fragmentos relevantes.")

    # Paso B: Generate
    print("   🤖 Generando respuesta...\n")
    
    prompt = f"""Usa el siguiente contexto para responder a la pregunta.
    
    Contexto:
    {context_text}
    
    Pregunta: {PREGUNTA}
    
    Respuesta:"""

    print("-" * 50)
    async for token in llm.stream(prompt):
        print(token, end="", flush=True)
    print("\n" + "-" * 50)

    # ==========================================
    # 4. LIMPIEZA
    # ==========================================
    print("\n4️⃣  Limpiando base de datos...")
    ids = [c.id for c in chunks]
    await vectorstore.delete(ids)
    print("   ✅ Limpieza terminada.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())