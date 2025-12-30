from typing import List, AsyncGenerator
from .advanced_hybrid import AdvancedHybridRAG  # ✅ Cambio: Importamos la versión avanzada
from srag.core import Chunk

class ModularRAG(AdvancedHybridRAG):
    """
    Estrategia RAG Modular (Actualizada con Advanced RAG).
    
    Orquesta múltiples módulos basándose en configuración:
    
    Flow:
    1. [Adaptive] Clasificación: ¿Necesito RAG? -> Si no, responde directo.
    2. [HyDE] Transformación (Opcional): Genera documento hipotético.
    3. [Advanced Hybrid] Recuperación: 
       - Multi-Query Expansion (si no se usa HyDE).
       - Búsqueda Vectorial Masiva.
       - Cross-Encoder Reranking.
    4. [Generation] Respuesta final.
    """

    def __init__(self, llm, embedder, vector_store, 
                use_adaptive: bool = True,
                use_hyde: bool = False, 
                use_hybrid: bool = True,
                reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"): # ✅ Nuevo param
        
        # Inicializamos la clase padre (AdvancedHybridRAG) que carga el reranker
        super().__init__(llm, embedder, vector_store, reranker_model=reranker_model)
        
        self.use_adaptive = use_adaptive
        self.use_hyde = use_hyde
        self.use_hybrid = use_hybrid

    async def _classify_query(self, query: str) -> bool:
        """Módulo Adaptive: Decide si buscar o no."""
        # Mantenemos la lógica original que funcionaba bien
        print("🤔 [Modular] Analizando intención de la pregunta...")
        
        prompt = f"""Eres un clasificador de consultas para un sistema RAG. Tu única tarea es decidir si la pregunta del usuario requiere buscar información en la base de datos documental.

        Criterios para BUSCAR:
        - Preguntas sobre documentos, archivos, textos específicos.
        - Preguntas técnicas, definiciones, resúmenes.
        - Preguntas sobre "el texto", "el documento", "hitos", "arquitectura".

        Criterios para CHATEAR:
        - Saludos (Hola, buenos días).
        - Preguntas personales al bot (¿Quién eres?, ¿Estás bien?).
        - Preguntas generales fuera del contexto (¿Cuánto es 2+2?).

        Pregunta: "{query}"

        Responde SOLO con una palabra: "BUSCAR" o "CHATEAR"."""
        
        resp = await self.llm.generate(prompt)
        clean_resp = resp.strip().upper()
        should_search = "BUSCAR" in clean_resp
        
        print(f"   -> Router LLM dijo: '{clean_resp}'")
        return should_search

    async def _generate_hyde_doc(self, query: str) -> str:
        """Módulo HyDE: Genera documento hipotético."""
        print("👻 [Modular] Generando alucinación hipotética (HyDE)...")
        prompt = f"""Escribe un breve párrafo técnico que responda idealmente a: "{query}". Inventa los datos si es necesario."""
        fake_doc = await self.llm.generate(prompt)
        return fake_doc

    async def retrieve(self, query: str, k: int = 4, **kwargs) -> List[Chunk]:
        search_query = query

        # 1. Módulo HyDE (Query Transformation)
        # Nota: Si activamos HyDE, la "Expansion" del AdvancedHybrid se hará sobre
        # el documento hipotético, lo cual puede ser muy potente o redundante.
        if self.use_hyde:
            fake_doc = await self._generate_hyde_doc(query)
            search_query = fake_doc
        
        # 2. Módulo de Recuperación (Advanced Hybrid o Simple)
        if self.use_hybrid:
            # ✅ Llama al retrieve de AdvancedHybridRAG (Expansion + Vector + Rerank)
            return await super().retrieve(search_query, k=k)
        else:
            # Fallback a búsqueda vectorial simple (sin rerank, sin expansion)
            print(f"🔍 [Modular] Búsqueda Vectorial Simple: '{search_query[:50]}...'")
            q_vec = await self.embedder.embed_query(search_query)
            return await self.vector_store.search(q_vec, k=k)

    async def stream(self, query: str, k: int = 4, **kwargs) -> AsyncGenerator[str, None]:
        # El flujo principal se mantiene igual, la magia ocurre dentro de retrieve()
        
        # 1. Módulo Adaptive
        if self.use_adaptive:
            needs_rag = await self._classify_query(query)
            if not needs_rag:
                print("⚡ [Modular] Modo Chat Directo")
                async for token in self.llm.stream(query):
                    yield token
                return

        # 2. Recuperación Avanzada
        chunks = await self.retrieve(query, k=k)
        
        if not chunks:
            yield "No encontré información relevante."
            return

        # 3. Generación
        context = self._build_context(chunks)
        print("🤖 [Modular] Generando respuesta final...")
        
        prompt = f"""Usa el siguiente contexto recuperado y reordenado para responder.
        
CONTEXTO:
{context}

PREGUNTA ORIGINAL: {query}
RESPUESTA:"""
        
        async for token in self.llm.stream(prompt):
            yield token