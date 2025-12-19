import pytest
from srag.core import Document
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
    """
    Prueba que los parámetros se guarden correctamente.
    El __init__ sigue siendo síncrono, así que este test no requiere async.
    """
    chunker = FixedLengthChunker(chunk_size=100, overlap=20)
    assert chunker.chunk_size == 100
    assert chunker.overlap == 20

@pytest.mark.asyncio
async def test_split_short_document(simple_doc):
    """
    Caso: El documento es más corto que el chunk_size.
    Resultado esperado: 1 solo chunk idéntico al original.
    """
    chunker = FixedLengthChunker(chunk_size=200, overlap=0)
    # CORRECCIÓN: Usamos await porque split ahora es async
    chunks = await chunker.split([simple_doc])
    
    assert len(chunks) == 1
    assert chunks[0].content == simple_doc.content
    assert chunks[0].metadata["source"] == "frase.txt"
    assert chunks[0].metadata["original_start"] == 0
    assert chunks[0].metadata["original_end"] == len(simple_doc.content)

@pytest.mark.asyncio
async def test_split_exact_overlap():
    """
    Caso: División con solapamiento (Overlap).
    Texto: "0123456789" (10 chars)
    Size: 5
    Overlap: 2
    """
    text = "0123456789"
    doc = Document(content=text, metadata={"id": "doc1"})
    
    chunker = FixedLengthChunker(chunk_size=5, overlap=2)
    chunks = await chunker.split([doc])
    
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

@pytest.mark.asyncio
async def test_metadata_preservation():
    """Verifica que los metadatos del documento padre se copien al hijo."""
    doc = Document(
        content="ABC", 
        metadata={"source": "file.pdf", "page": 10, "hidden": True}
    )
    chunker = FixedLengthChunker(chunk_size=10, overlap=0)
    chunks = await chunker.split([doc])
    
    chunk_meta = chunks[0].metadata
    
    assert chunk_meta["source"] == "file.pdf"
    assert "source" in chunk_meta
    assert chunk_meta["original_start"] == 0

@pytest.mark.asyncio
async def test_multiple_documents():
    """Prueba procesando una lista de varios documentos."""
    docs = [
        Document(content="A" * 10, metadata={"id": 1}),
        Document(content="B" * 10, metadata={"id": 2})
    ]
    chunker = FixedLengthChunker(chunk_size=6, overlap=0)
    
    chunks = await chunker.split(docs)
    
    assert len(chunks) == 4
    assert chunks[0].content == "AAAAAA"
    assert chunks[1].content == "AAAA"
    assert chunks[2].content == "BBBBBB"
    assert chunks[3].content == "BBBB"

@pytest.mark.asyncio
async def test_empty_input():
    """Prueba con lista vacía."""
    chunker = FixedLengthChunker()
    # Comparar el resultado de await con la lista vacía
    result = await chunker.split([])
    assert result == []

def test_large_overlap_error_prevention():
    """
    Test placeholder para futura validación de lógica.
    No requiere async ya que no estamos llamando a split aquí.
    """
    pass