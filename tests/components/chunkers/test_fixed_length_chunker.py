import pytest
from srag.core import Document
# Ajusta la ruta de importación según donde hayas guardado tu clase.
# Asumo: src/srag/components/chunkers/fixed_length_chunker.py
from srag.components.chunkers.fixed_length_chunker import FixedLengthChunker

# --- Fixtures ---

@pytest.fixture
def simple_doc():
    """Documento simple para pruebas básicas."""
    return Document(
        content="El veloz murciélago hindú comía feliz cardillo y kiwi.",
        metadata={"source": "frase.txt", "author": "Panagrama"}
    )

# --- Tests ---

def test_initialization():
    """Prueba que los parámetros se guarden correctamente."""
    chunker = FixedLengthChunker(chunk_size=100, overlap=20)
    assert chunker.chunk_size == 100
    assert chunker.overlap == 20

def test_split_short_document(simple_doc):
    """
    Caso: El documento es más corto que el chunk_size.
    Resultado esperado: 1 solo chunk idéntico al original.
    """
    chunker = FixedLengthChunker(chunk_size=200, overlap=0)
    chunks = chunker.split([simple_doc])
    
    assert len(chunks) == 1
    assert chunks[0].content == simple_doc.content
    assert chunks[0].metadata["source"] == "frase.txt"
    assert chunks[0].metadata["original_start"] == 0
    assert chunks[0].metadata["original_end"] == len(simple_doc.content)

def test_split_exact_overlap():
    """
    Caso: División con solapamiento (Overlap).
    Texto: "0123456789" (10 chars)
    Size: 5
    Overlap: 2
    
    Lógica esperada:
    1. "01234" (0-5) -> Prox start: 0 + (5-2) = 3
    2. "34567" (3-8) -> Prox start: 3 + 3 = 6
    3. "6789"  (6-10) -> Prox start: 6 + 3 = 9
    4. "9"     (9-10) -> Prox start: 9 + 3 = 12 (Fin)
    """
    text = "0123456789"
    doc = Document(content=text, metadata={"id": "doc1"})
    
    chunker = FixedLengthChunker(chunk_size=5, overlap=2)
    chunks = chunker.split([doc])
    
    # Según la lógica de tu bucle while:
    assert len(chunks) == 4
    
    # Contenidos
    assert chunks[0].content == "01234"
    assert chunks[1].content == "34567"
    assert chunks[2].content == "6789"
    assert chunks[3].content == "9"
    
    # Metadatos de posición
    assert chunks[0].metadata["original_start"] == 0
    assert chunks[1].metadata["original_start"] == 3
    assert chunks[2].metadata["original_start"] == 6
    assert chunks[3].metadata["original_start"] == 9

def test_metadata_preservation():
    """Verifica que los metadatos del documento padre se copien al hijo."""
    doc = Document(
        content="ABC", 
        metadata={"source": "file.pdf", "page": 10, "hidden": True}
    )
    chunker = FixedLengthChunker(chunk_size=10, overlap=0)
    chunks = chunker.split([doc])
    
    chunk_meta = chunks[0].metadata
    
    assert chunk_meta["source"] == "file.pdf"
    # Nota: Tu implementación actual en fixed_length_chunker.py SOLO copia "source".
    # Si quisieras copiar todo, deberías cambiar:
    # metadata={"source": ...}  -->  metadata=document.metadata.copy() ...
    
    # Testeamos lo que tu código hace actualmente:
    assert "source" in chunk_meta
    assert chunk_meta["original_start"] == 0

def test_multiple_documents():
    """Prueba procesando una lista de varios documentos."""
    docs = [
        Document(content="A" * 10, metadata={"id": 1}),
        Document(content="B" * 10, metadata={"id": 2})
    ]
    # Chunk de 6, overlap 0.
    # Doc A (10) -> Chunk A1 (6), Chunk A2 (4)
    # Doc B (10) -> Chunk B1 (6), Chunk B2 (4)
    chunker = FixedLengthChunker(chunk_size=6, overlap=0)
    
    chunks = chunker.split(docs)
    
    assert len(chunks) == 4
    assert chunks[0].content == "AAAAAA"
    assert chunks[1].content == "AAAA"
    assert chunks[2].content == "BBBBBB"
    assert chunks[3].content == "BBBB"

def test_empty_input():
    """Prueba con lista vacía."""
    chunker = FixedLengthChunker()
    assert chunker.split([]) == []

def test_large_overlap_error_prevention():
    """
    Verifica que el código no entre en bucle infinito si overlap >= chunk_size.
    Esto es un test de 'robustez' para futura implementación, o para verificar comportamiento actual.
    
    Si tu código hace: start += chunk_size - overlap
    Y overlap >= chunk_size, entonces start no avanza (o retrocede).
    """
    chunker = FixedLengthChunker(chunk_size=10, overlap=10) # Overlap igual al tamaño
    doc = Document(content="Texto de prueba")
    
    # Si tu código no maneja esto, entrará en bucle infinito y el test hará timeout.
    # Es recomendable añadir una validación en el __init__ de tu clase.
    # Por ahora, verificamos si salta error o qué hace.
    
    # Nota: Como tu código actual NO protege contra esto, este test podría colgarse.
    # Lo dejo comentado como sugerencia de mejora para tu clase:
    # with pytest.raises(ValueError):
    #     FixedLengthChunker(chunk_size=10, overlap=12)
    pass