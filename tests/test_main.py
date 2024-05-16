import sys
import os
from io import BytesIO
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, mock_open

import pytest
from fastapi.testclient import TestClient
from main import app, MinioFunctions, chroma_client, change_extension_to_json, create_collection, embedding_func, \
    OCR_SIM_DIRECTORY

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

client = TestClient(app)


@pytest.fixture
def mock_ocr_data():
    return {
        "filename": "example.pdf",
        "content": '{"analyzeResult": {"content": "This is a sample OCR text\\nWith multiple lines\\nAnd more '
                   'content"}} '
    }


@pytest.fixture
def mock_minio_functions():
    with patch.object(MinioFunctions, '__init__', lambda x, y: None):
        minio_functions = MinioFunctions("yourbucket")
        minio_functions.minioClient = MagicMock()
        minio_functions.check_file_uploaded = MagicMock(return_value=True)
        return minio_functions


@pytest.fixture
def mock_chroma_client():
    with patch.object(chroma_client, 'get_collection') as mock_get_collection:
        mock_collection = MagicMock()
        mock_collection.get = MagicMock(return_value={'data': None,
                                                      'documents': ['This is a sample OCR text',
                                                                    'With multiple lines',
                                                                    'And more content'],
                                                      'embeddings': None,
                                                      'ids': ['example.pdf_0', 'example.pdf_1', 'example.pdf_2'],
                                                      'metadatas': [{'filename': 'example.pdf'},
                                                                    {'filename': 'example.pdf'},
                                                                    {'filename': 'example.pdf'}],
                                                      'uris': None})
        mock_get_collection.return_value = mock_collection
        yield mock_get_collection


@pytest.fixture
def mock_chroma_client_create():
    with patch('main.chroma_client', autospec=True) as mock_client:
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        yield mock_client


@pytest.mark.asyncio
async def test_create_collection(mock_ocr_data, mock_chroma_client_create):
    filename = mock_ocr_data["filename"]
    file_content = mock_ocr_data["content"]
    # Mocking file path to JSON data
    # Determine the expected file path based on the environment
    environment = os.getenv("ENVIRONMENT", "local")
    if environment == "docker":
        file_path = OCR_SIM_DIRECTORY / change_extension_to_json(filename)
    else:
        file_path = Path(__file__).resolve().parent.parent / "OCR_data" / change_extension_to_json(filename)

    m_open = mock_open(read_data=file_content)
    with patch('builtins.open', m_open):
        await create_collection(file_path, filename)

        # Check if the file was opened correctly by looking at all calls
        m_open.assert_any_call(file_path, 'r', encoding='utf8')

        # Verify that the collection was created or retrieved with the correct name and embedding function
        collection_name = filename.replace(" ", "")
        mock_chroma_client_create.get_or_create_collection.assert_called_once_with(
            name=collection_name,
            embedding_function=embedding_func  # Ensure we reference embedding_func correctly
        )

        # Check if documents were added to the collection correctly
        created_collection = mock_chroma_client_create.get_or_create_collection.return_value
        created_collection.add.assert_called_once()
        add_call_args = created_collection.add.call_args[1]
        assert add_call_args['documents'] == [
            "This is a sample OCR text", "With multiple lines", "And more content"
        ]
        assert add_call_args['ids'] == [
            f"{collection_name}_0", f"{collection_name}_1", f"{collection_name}_2"
        ]
        assert add_call_args['metadatas'] == [
            {"filename": filename}, {"filename": filename}, {"filename": filename}
        ]


def test_upload_endpoint(mock_minio_functions):
    with patch('main.minioFunctions', mock_minio_functions):
        files = [
            ('files', ('example1.pdf', b'file content here', 'application/pdf')),
            ('files', ('example2.png', b'file content here', 'image/png'))
        ]
        response = client.post("/upload", files=files)
        assert response.status_code == 200
        json_response = response.json()
        assert len(json_response) == 2
        for res in json_response:
            assert "filename" in res
            assert "status" in res
            assert res["status"] == "success"


@pytest.fixture
def mock_minio_functions_ocr():
    with patch.object(MinioFunctions, 'check_file_uploaded', return_value=True):
        yield MinioFunctions("test-bucket")


@pytest.mark.asyncio
async def test_ocr_endpoint(mock_ocr_data, mock_minio_functions, mock_minio_functions_ocr):
    filename = mock_ocr_data["filename"]
    file_content = mock_ocr_data["content"]

    # Mocking file path to JSON data
    # Determine the expected file path based on the environment
    environment = os.getenv("ENVIRONMENT", "local")
    if environment == "docker":
        file_path = OCR_SIM_DIRECTORY / change_extension_to_json(filename)
    else:
        file_path = Path(__file__).resolve().parent.parent / "OCR_data" / change_extension_to_json(filename)

    m_open = mock_open(read_data=file_content)
    with patch('builtins.open', m_open) as mock_file:
        # Ensure the file path exists
        with patch.object(Path, 'is_file', return_value=True):
            response = client.post("/ocr", json={"filename": filename})
            assert response.status_code == 200
            assert response.json() == {"message": "Imported Simulated OCR data to Vector DB."}

            # Check if the file was opened correctly by looking at all calls
            mock_file.assert_any_call(file_path, 'r', encoding='utf8')


def test_ocr_endpoint_with_invalid_data():
    response = client.post("/ocr", json={"filename": "invalid_file.txt"})
    assert response.status_code == 404
    assert response.json()["detail"] == "File not found"


def test_extract_endpoint():
    response = client.post("/extract", json={"filename": "example.pdf", "query": "sample query"})
    assert response.status_code == 200
    assert "response" in response.json()


def test_extract_endpoint_with_missing_file():
    response = client.post("/extract", json={"filename": "nonexistent.pdf", "query": "sample query"})
    assert response.status_code == 404
    assert response.json()["detail"] == "File has not been put into the vector database."


def test_list_collections_endpoint():
    response = client.get("/list_collections")
    assert response.status_code == 200


def test_list_documents_endpoint(mock_chroma_client):
    response = client.get("/list_documents?filename=example.pdf")
    assert response.status_code == 200
    assert response.json() == {'data': None,
                               'documents': ['This is a sample OCR text',
                                             'With multiple lines',
                                             'And more content'],
                               'embeddings': None,
                               'ids': ['example.pdf_0', 'example.pdf_1', 'example.pdf_2'],
                               'metadatas': [{'filename': 'example.pdf'},
                                             {'filename': 'example.pdf'},
                                             {'filename': 'example.pdf'}],
                               'uris': None}

    # Ensure the collection was accessed with the correct name
    mock_chroma_client.assert_called_once_with("example.pdf".replace(" ", ""))


def test_reset_db_endpoint():
    response = client.post("/reset")
    assert response.status_code == 200
    assert response.json() == {"message": "ChromaDB has been reset, all collections and documents have been removed."}


@pytest.mark.asyncio
async def test_minio_upload_file_direct(mock_minio_functions):
    mock_file = MagicMock()
    mock_file.read = AsyncMock(return_value=b"file content")
    mock_file.size = 1024
    mock_file.filename = "example.pdf"

    minio_functions = MinioFunctions("yourbucket")
    minio_functions.minioClient = MagicMock()

    with patch.object(minio_functions.minioClient, 'put_object', return_value=None) as mock_put_object:
        response = await minio_functions.upload_file(mock_file)
        assert response["message"] == "File uploaded successfully: example.pdf"
        mock_put_object.assert_called_once()

        # Validate the parameters of the call to put_object
        args, kwargs = mock_put_object.call_args
        assert args[0] == "yourbucket"
        assert args[1] == "example.pdf"
        assert isinstance(args[2], BytesIO)
        assert args[2].getvalue() == b"file content"  # Ensure the content matches
        assert args[3] == 1024


def test_minio_check_file_uploaded(mock_minio_functions):
    with patch('main.minioFunctions', mock_minio_functions):
        mock_minio_functions.minioClient.stat_object = MagicMock(return_value=True)
        assert mock_minio_functions.check_file_uploaded("example.pdf") == True
