from typing import List, AsyncGenerator, Set
from .base import BaseRAGStrategy
from srag.core import Chunk
from .hybrid import HybridRAG  # Reutilizamos la lógica de Hybrid y RRF

class ModularRAG(HybridRAG):
    """
    Estrategia RAG Modular (State of the Art).
    Orquesta múltiples módulos basándose en configuración:
    
    Flow:
    1. [Adaptive] Clasificación: ¿Necesito RAG? -> Si no, responde directo.
    2. [HyDE] Transformación: ¿Genero documento hipotético para mejorar búsqueda?
    3. [Hybrid] Recuperación: Vectores + Keywords + RRF Fusion.
    4. [Generation] Respuesta final.
    """

    def __init__(self, llm, embedder, vector_store, 
                use_adaptive: bool = True,
                use_hyde: bool = False, 
                use_hybrid: bool = True):
        super().__init__(llm, embedder, vector_store)
        self.use_adaptive = use_adaptive
        self.use_hyde = use_hyde
        self.use_hybrid = use_hybrid

    async def _classify_query(self, query: str) -> bool:
        """Módulo Adaptive: Decide si buscar o no (Versión Robusta)."""
        print("🤔 [Modular] Analizando intención de la pregunta...")
        
        # PROMPT MEJORADO (Igual que en AdaptiveRAG que funcionó bien)
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
        
        # Usamos generate para obtener la decisión
        resp = await self.llm.generate(prompt)
        clean_resp = resp.strip().upper()
        
        # Lógica de decisión más permisiva
        should_search = "BUSCAR" in clean_resp
        
        print(f"   -> Router LLM dijo: '{clean_resp}'")
        print(f"   -> Decisión final: {'✅ Requiere RAG' if should_search else '⚡ Conversación directa'}")
        
        return should_search

    async def _generate_hyde_doc(self, query: str) -> str:
        """Módulo HyDE: Genera documento hipotético."""
        print("👻 [Modular] Generando alucinación hipotética (HyDE)...")
        prompt = f"""Escribe un breve párrafo técnico que responda idealmente a: "{query}". Inventa los datos si es necesario."""
        fake_doc = await self.llm.generate(prompt)
        return fake_doc

    async def retrieve(self, query: str, k: int = 4, **kwargs) -> List[Chunk]:
        vector_search_text = query

        # 1. Módulo HyDE: Transformamos SOLO la query vectorial
        if self.use_hyde:
            fake_doc = await self._generate_hyde_doc(query)
            vector_search_text = fake_doc  # La alucinación
            print("   👻 HyDE: Usando documento hipotético para búsqueda vectorial.")
        
        # 2. Módulo Híbrido
        if self.use_hybrid:
            # MAGIA: Pasamos la alucinación para el vector, 
            # pero HybridRAG usará 'query' (original) para las palabras clave.
            return await super().retrieve(
                query=query,          # Original para Keywords
                vector_query=vector_search_text, # Alucinación para Vectores
                k=k
            )
        else:
            # Fallback simple
            q_vec = await self.embedder.embed_query(vector_search_text)
            return await self.vector_store.search(q_vec, k=k)

    async def stream(self, query: str, k: int = 4, **kwargs) -> AsyncGenerator[str, None]:
        # 1. Módulo Adaptive (Pre-retrieval)
        if self.use_adaptive:
            needs_rag = await self._classify_query(query)
            if not needs_rag:
                print("⚡ [Modular] Modo Chat Directo (Fast path)")
                async for token in self.llm.stream(query):
                    yield token
                return

        # 2. Recuperación (Incluye HyDE y Hybrid si están activos)
        chunks = await self.retrieve(query, k=k)
        
        if not chunks:
            yield "No encontré información relevante."
            return

        # 3. Generación Final
        context = self._build_context(chunks)
        print("🤖 [Modular] Generando respuesta final...")
        
        prompt = f"""Usa el contexto proporcionado para responder.
        
CONTEXTO:
{context}

PREGUNTA ORIGINAL: {query}
RESPUESTA:"""
        
        async for token in self.llm.stream(prompt):
            yield token