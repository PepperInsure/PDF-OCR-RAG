import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from minio.error import S3Error
from file_server_utils import MinioFunctions


@pytest.fixture
def mock_minio_functions():
    with patch.object(MinioFunctions, '__init__', lambda x, y: None):
        minio_functions = MinioFunctions("yourbucket")
        minio_functions.minioClient = MagicMock()
        return minio_functions


def test_initialize_bucket(mock_minio_functions):
    with patch.object(mock_minio_functions.minioClient, 'make_bucket', return_value=None) as mock_make_bucket:
        mock_minio_functions.initialize_bucket()
        mock_make_bucket.assert_called_once_with(mock_minio_functions.bucket_name)


def test_initialize_bucket_bucket_exists(mock_minio_functions):
    error = S3Error(
        code="BucketAlreadyOwnedByYou", message="", resource="", request_id="", host_id="", response=None
    )
    with patch.object(mock_minio_functions.minioClient, 'make_bucket', side_effect=error) as mock_make_bucket:
        mock_minio_functions.initialize_bucket()
        mock_make_bucket.assert_called_once_with(mock_minio_functions.bucket_name)


@pytest.mark.asyncio
async def test_upload_file(mock_minio_functions):
    mock_file = MagicMock()
    mock_file.read = AsyncMock(return_value=b"file content")
    mock_file.size = 1024
    mock_file.filename = "example.pdf"
    with patch.object(mock_minio_functions.minioClient, 'put_object', return_value=None):
        with patch.object(mock_minio_functions, 'check_file_uploaded', return_value=True):
            response = await mock_minio_functions.upload_file(mock_file)
            assert response == {"message": "File uploaded successfully: example.pdf"}


@pytest.mark.asyncio
async def test_upload_file_failure(mock_minio_functions):
    mock_file = MagicMock()
    mock_file.read = AsyncMock(return_value=b"file content")
    mock_file.size = 1024
    mock_file.filename = "example.pdf"
    error = S3Error(
        code="InternalError", message="", resource="", request_id="", host_id="", response=None
    )
    with patch.object(mock_minio_functions.minioClient, 'put_object', side_effect=error):
        response = await mock_minio_functions.upload_file(mock_file)
        assert response["message"] == "Failed to upload file"
        assert "error" in response


def test_check_file_uploaded(mock_minio_functions):
    with patch.object(mock_minio_functions.minioClient, 'stat_object', return_value=True):
        assert mock_minio_functions.check_file_uploaded("example.pdf") == True  # noqa: E712


def test_check_file_uploaded_not_found(mock_minio_functions):
    error = S3Error(
        code="NoSuchKey", message="", resource="", request_id="", host_id="", response=None
    )
    with patch.object(mock_minio_functions.minioClient, 'stat_object', side_effect=error):
        assert mock_minio_functions.check_file_uploaded("example.pdf") == False  # noqa: E712
