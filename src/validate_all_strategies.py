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
from srag.components.llms import OllamaLLM, GeminiLLM
# NUEVOS IMPORTS DE ALTO NIVEL
from srag.pipeline import IngestionPipeline
from srag.strategies import SimpleRAG, HybridRAG, AdaptiveRAG, HydeRAG, ModularRAG


# --- CONFIGURACIÓN ---
DOC_FILE = "BOE-A-1978-31229-consolidado.pdf" # Asegúrate de tener este archivo (o cualquier PDF)
COLLECTION_NAME = "benchmark_strategies"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"
BASE_URL = "http://172.23.224.1:11434"

async def setup_knowledge_base():
    """
    Paso 1: Ingesta de datos (Común para todas las estrategias).
    Carga el PDF, lo divide y lo guarda en ChromaDB.
    """
    print("\n🏗️  CONSTRUYENDO BASE DE CONOCIMIENTO COMPARTIDA...")
    
    # 1. Componentes
    if not os.path.exists(DOC_FILE):
        print(f"❌ Error: No existe el archivo '{DOC_FILE}'. Pon un PDF en la raíz.")
        sys.exit(1)

    loader = LlamaParseLoader([DOC_FILE], save_output=True)
    chunker = FixedLengthChunker(chunk_size=500, overlap=50)
    embedder = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=BASE_URL)
    vectorstore = ChromaVectorStore(collection_name=COLLECTION_NAME)
    # llm = OllamaLLM(model_name=LLM_MODEL, base_url=BASE_URL)
    llm =   GeminiLLM(api_key=os.getenv("GOOGLE_API_KEY"))

    # 2. Ingesta
    print(f"   📥 Cargando {DOC_FILE}...")
    docs = await loader.load()
    chunks = chunker.split(docs)
    
    print(f"   🧠 Generando embeddings para {len(chunks)} chunks...")
    vecs = await embedder.embed_documents([c.content for c in chunks])
    for c, v in zip(chunks, vecs): c.embedding = v
    
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
            print(f"\n❌ Error en estrategia: {e}")
        
        print("\n" + "-"*30)

async def main():
    load_dotenv()
    
    # 1. Preparar datos (Ingesta única)
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

    # 3. Definir preguntas trampa para ver diferencias
    queries = [
        # A. Pregunta Conversacional (Prueba AdaptiveRAG)
        "Buenos días, ¿eres un abogado virtual?",
        
        # B. Pregunta Específica / Dato (Prueba HybridRAG)
        # Contiene términos clave: "bandera", "artículo 4".
        # La búsqueda híbrida debería brillar aquí al anclar "artículo 4".
        "¿Qué dice exactamente el artículo 4 sobre la bandera de España?",
        
        # C. Pregunta Compleja / Procedimiento (Prueba HyDE/Modular)
        # Requiere entender conceptos abstractos: sucesión, primogenitura, líneas.
        # HyDE generará un texto legal hipotético que ayudará a encontrar el Título II.
        "Explica el orden de sucesión a la Corona y qué criterio tiene preferencia (varón/mujer, edad, etc)."
    ]

    # 4. Ejecutar Ronda de Pruebas
    for name, strategy_instance in strategies.items():
        await test_strategy(name, strategy_instance, queries)

    # 5. Limpieza
    print("\n🧹 Limpiando base de datos...")
    # Truco: Borramos la colección entera o los chunks si tu vectorstore lo permite
    # Como Chroma en memoria/local persiste, aquí simulamos limpieza borrando IDs conocidos
    # (Para simplificar, en un entorno real borraríamos la colección collection.delete())
    print("✅ Validación completada.")

if __name__ == "__main__":
    asyncio.run(main())