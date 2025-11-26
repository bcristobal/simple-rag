import pytest
from unittest.mock import patch, MagicMock, AsyncMock, mock_open
from srag.components.loaders.llama_parse_loader import LlamaParseLoader
from srag.core import Document

# --- Fixtures ---

@pytest.fixture
def mock_llama_parse_class():
    """Mockea la clase LlamaParse para evitar llamadas a la API."""
    # Apuntamos al archivo donde se importa LlamaParse
    with patch('srag.components.loaders.llama_parse_loader.LlamaParse') as MockClass:
        mock_instance = MockClass.return_value
        # Configuramos aload_data como asíncrono
        mock_instance.aload_data = AsyncMock()
        yield MockClass, mock_instance

@pytest.fixture
def mock_pymupdf():
    """Mockea pymupdf para evitar leer archivos reales."""
    with patch('srag.components.loaders.llama_parse_loader.pymupdf') as mock_fitz:
        yield mock_fitz

# --- Tests ---

def test_init_config(mock_llama_parse_class):
    """Verifica que se pasa la configuración correcta a LlamaParse."""
    MockClass, _ = mock_llama_parse_class
    
    # Caso 1: Pasando api_key explícita
    loader = LlamaParseLoader(file_paths=[])
    
    # Verificar llamada al constructor de LlamaParse
    _, kwargs = MockClass.call_args
    assert kwargs["result_type"] == "markdown"

@pytest.mark.asyncio
async def test_load_success(mock_llama_parse_class, mock_pymupdf):
    """Prueba el flujo normal de carga."""
    _, mock_parser_instance = mock_llama_parse_class
    
    # 1. Simular respuesta de LlamaParse (Lista de objetos LlamaIndex)
    mock_doc = MagicMock()
    mock_doc.text = "# Título\nContenido extraído."
    mock_doc.metadata = {"page_label": "1"}
    
    mock_parser_instance.aload_data.return_value = [mock_doc]

    # 2. Simular sistema de archivos (os.path.exists siempre True)
    with patch('os.path.exists', return_value=True):
        # 3. Simular PyMuPDF para el ID estable
        mock_doc_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Contenido original del PDF"
        mock_doc_pdf.__enter__.return_value = [mock_page] # Iteramos sobre páginas
        mock_pymupdf.open.return_value = mock_doc_pdf
        
        # EJECUCIÓN
        loader = LlamaParseLoader(file_paths=["dummy.pdf"])
        documents = await loader.load()

    # 4. Verificaciones
    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert documents[0].content == "# Título\nContenido extraído."
    assert documents[0].metadata["source"] == "dummy.pdf"
    assert documents[0].metadata["loader"] == "LlamaParse"
    
    # Verificar que se llamó al parser con el archivo
    mock_parser_instance.aload_data.assert_called_once_with("dummy.pdf")

@pytest.mark.asyncio
async def test_save_markdown(mock_llama_parse_class, mock_pymupdf):
    """Verifica la lógica de guardado de backup .md."""
    _, mock_parser_instance = mock_llama_parse_class
    
    # Configurar respuesta dummy
    mock_doc = MagicMock()
    mock_doc.text = "Markdown content"
    mock_doc.metadata = {}
    mock_parser_instance.aload_data.return_value = [mock_doc]

    # Mockeamos 'open' para interceptar la escritura en disco
    m_open = mock_open()
    
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', m_open):
            # Mock básico de pymupdf
            mock_doc_pdf = MagicMock()
            mock_doc_pdf.__enter__.return_value = []
            mock_pymupdf.open.return_value = mock_doc_pdf

            # Instanciamos con save_output=True
            loader = LlamaParseLoader(file_paths=["report.pdf"], save_output=True)
            await loader.load()

    # Verificar que se intentó abrir 'report.pdf.md' en modo escritura ('w')
    m_open.assert_called_with("report.pdf.md", "w", encoding="utf-8")
    
    # Verificar que se escribió el contenido
    handle = m_open()
    handle.write.assert_called_with("Markdown content")

@pytest.mark.asyncio
async def test_file_not_found(mock_llama_parse_class):
    """Prueba que el loader ignora archivos inexistentes sin romper la ejecución."""
    with patch('os.path.exists', return_value=False):
        loader = LlamaParseLoader(file_paths=["no_existe.pdf"])
        docs = await loader.load()
        
        assert docs == []
        # No se debe intentar parsear nada
        _, mock_parser = mock_llama_parse_class
        mock_parser.aload_data.assert_not_called()