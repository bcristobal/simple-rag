import asyncio
import sys
import os
import time  # <--- Nuevo: Para controlar el Rate Limit
from dotenv import load_dotenv

# Aseguramos que Python encuentre el paquete 'srag'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importamos componentes
from srag.components.loaders import LlamaParseLoader
# from srag.components.chunkers.fixed_length_chunker import FixedLengthChunker  <-- ELIMINADO
from srag.components.chunkers.hybrid_chunker import HybridChunker # <-- NUEVO
from srag.components.embeddings import OllamaEmbeddings
from srag.components.vectorstores import ChromaVectorStore
from srag.components.llms import OllamaLLM, GeminiLLM

# Imports de estrategias
from srag.strategies import SimpleRAG, HybridRAG, AdaptiveRAG, HydeRAG, ModularRAG

# --- CONFIGURACIÓN ---
DOC_FILE = "BOE-A-1978-31229-consolidado.pdf"
COLLECTION_NAME = "benchmark_strategies"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "gemini-2.0-flash" # O el modelo que uses
BASE_URL = "http://172.23.224.1:11434" # Ajusta a tu IP si usas Ollama remoto
OLLAMA_MODEL = "phi4-mini:3.8b"

async def setup_knowledge_base():
    """
    Paso 1: Ingesta de datos (Común para todas las estrategias).
    Carga el PDF, lo divide con HybridChunker y lo guarda en ChromaDB.
    """
    print("\n🏗️  CONSTRUYENDO BASE DE CONOCIMIENTO COMPARTIDA...")
    
    # 1. Verificación de archivo
    if not os.path.exists(DOC_FILE):
        print(f"❌ Error: No existe el archivo '{DOC_FILE}'. Pon un PDF en la raíz.")
        sys.exit(1)

    # 2. Instanciación de Componentes (ORDEN IMPORTANTE)
    loader = LlamaParseLoader([DOC_FILE], save_output=True)
    
    # Instanciamos Embedder y LLM primero porque el Chunker los necesita
    embedder = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=BASE_URL)
    
    # Configura tu LLM (Gemini u Ollama)
    if os.getenv("GOOGLE_API_KEY"):
        llm = GeminiLLM(api_key=os.getenv("GOOGLE_API_KEY"))
    else:
        print("⚠️ No se detectó GOOGLE_API_KEY, usando Ollama...")
        llm = OllamaLLM(model_name=OLLAMA_MODEL, base_url=BASE_URL)

    print("   🔨 Inicializando HybridChunker (Estructural + Semántico)...")
    chunker = HybridChunker(
        embedder=embedder,
        percentile_threshold=90.0,
        min_chunk_size_words=20,    # Evita fragmentos basura
        max_chunk_size_words=500
    )

    vectorstore = ChromaVectorStore(collection_name=COLLECTION_NAME)

    # 3. Ingesta
    print(f"   📥 Cargando {DOC_FILE}...")
    docs = await loader.load()
    
    print("   🧩 Ejecutando división híbrida (esto puede tardar unos segundos)...")
    chunks = await chunker.split(docs)
    
    print(f"   🧠 Generando embeddings finales para {len(chunks)} chunks e indexando...")
    # Calculamos embeddings para la búsqueda vectorial
    vecs = await embedder.embed_documents([c.content for c in chunks])
    for c, v in zip(chunks, vecs): 
        c.embedding = v
    
    await vectorstore.add(chunks)
    print("   ✅ Base de datos lista.\n")
    
    return llm, embedder, vectorstore

async def test_strategy(name: str, strategy, queries: list):
    """Ejecuta una lista de preguntas contra una estrategia específica."""
    print(f"\n{'='*60}")
    print(f"🧪  PROBANDO ESTRATEGIA: {name}")
    print(f"{'='*60}")

    for q in queries:
        print(f"\n❓ Pregunta: '{q}'")
        print(f"   Respuesta: ", end="", flush=True)
        
        try:
            async for token in strategy.stream(q):
                print(token, end="", flush=True)
        except Exception as e:
            print(f"\n❌ Error en estrategia durante el stream: {e}")
        
        print("\n" + "-"*30)

async def main():
    load_dotenv()
    
    # 1. Preparar datos (Ingesta única con HybridChunker)
    llm, embedder, vectorstore = await setup_knowledge_base()

    # 2. Definir las estrategias a probar
    strategies = {
        "1. SimpleRAG (Baseline)": SimpleRAG(llm, embedder, vectorstore),
        
        "2. HybridRAG (Vector + Keywords)": HybridRAG(llm, embedder, vectorstore),
        
        "3. AdaptiveRAG (Router)": AdaptiveRAG(llm, embedder, vectorstore),
        
        "4. HyDE RAG (Alucinación)": HydeRAG(llm, embedder, vectorstore),
        
        "5. ModularRAG (ALL-IN)": ModularRAG(
            llm, embedder, vectorstore, 
            use_adaptive=True, use_hyde=True, use_hybrid=True
        )
    }

    # 3. Definir preguntas de prueba
    queries = [
        # A. Conversacional
        "Buenos días, ¿eres un abogado virtual?",
        
        # B. Dato específico (Debe encontrar "Artículo 4" gracias al nuevo chunker)
        "¿Qué dice exactamente el artículo 4 sobre la bandera de España?",
        
        # C. Compleja
        "Explica el orden de sucesión a la Corona y qué criterio tiene preferencia."
    ]

    # 4. Ejecutar Ronda de Pruebas con pausas
    for name, strategy_instance in strategies.items():
        try:
            await test_strategy(name, strategy_instance, queries)
        except Exception as e:
            print(f"❌ Error crítico ejecutando la estrategia {name}: {e}")
        
        print("\n⏳ Enfriando API de Google (15s) para evitar 'Quota Exceeded'...")
        time.sleep(15)

    # 5. Limpieza (Opcional)
    print("\n🧹 Limpieza finalizada.")
    print("✅ Validación completada.")

if __name__ == "__main__":
    asyncio.run(main())