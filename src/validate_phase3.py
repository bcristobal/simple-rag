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
# NUEVOS IMPORTS DE ALTO NIVEL
from srag.pipeline import IngestionPipeline
from srag.strategies import SimpleRAG, HybridRAG, AdaptiveRAG, HydeRAG, ModularRAG


# --- CONFIGURACIÓN ---
PDF_FILE = "Hito_1_VC-06.pdf"  # Asegúrate de que este archivo existe en la raíz o ruta correcta
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://172.23.224.1:11434"
COLLECTION_NAME = "fase3_pipeline_completo"

async def main():
    load_dotenv()  # Cargar variables de entorno desde .env si es necesario
    # 1. Configuración de Componentes (Piezas de Lego)
    loader = LlamaParseLoader([PDF_FILE], save_output=False)
    chunker = FixedLengthChunker(chunk_size=500, overlap=50)
    embedder = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    vectorstore = ChromaVectorStore(COLLECTION_NAME)
    llm = OllamaLLM(model_name=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    # 2. Fase de Ingesta (Automática)
    pipeline = IngestionPipeline(loader, chunker, embedder, vectorstore)
    await pipeline.run()

    # 3. Fase de Chat (Usando Estrategia)
    rag = SimpleRAG(llm, embedder, vectorstore)
    
    print("\n🤖 Chatbot listo. Escribe 'salir' para terminar.")
    while True:
        query = input("\nPregunta: ")
        if query.lower() == "salir": break
        
        print("Respuesta: ", end="", flush=True)
        async for token in rag.stream(query):
            print(token, end="", flush=True)
        print()

if __name__ == "__main__":
    asyncio.run(main())