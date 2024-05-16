import os
import io
import logging

from minio import Minio
from minio.error import S3Error

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
minio_endpoint = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
minio_access_key = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
minio_secret_key = os.getenv('MINIO_SECRET_KEY', 'minioadmin')


class MinioFunctions:
    """Class for handling Minio operations."""
    bucket_name = None

    def __init__(self, bucket_name):
        """
        Initialize Minio client and ensure the bucket exists.

        Parameters:
        bucket_name (str): The name of the bucket to use.
        """
        # Initialize a Minio client
        self.minioClient = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=False
        )
        self.bucket_name = bucket_name
        self.initialize_bucket()

    async def upload_file(self, file):
        """
        Upload a file to Minio.

        Parameters:
        file (UploadFile): The file to upload.

        Returns:
        dict: A message indicating the upload status.
        """
        try:
            file_content = await file.read()  # Read the file content into bytes
            file_stream = io.BytesIO(file_content)
            file_size = file.size  # Calculate the size of the file in bytes
            self.minioClient.put_object(
                self.bucket_name, file.filename, file_stream, file_size
            )
            file_uploaded = self.check_file_uploaded(file.filename)
            if file_uploaded:
                logger.info(f"File uploaded successfully: {file.filename}")
                return {"message": "File uploaded successfully: " + file.filename}
            else:
                logger.error(f"File upload failed: {file.filename}")
                return {"message": "File upload failed"}
        except S3Error as err:
            logger.error(f"Failed to upload file {file.filename}: {str(err)}")
            return {"message": "Failed to upload file", "error": str(err)}

    def initialize_bucket(self):
        """Ensure the Minio bucket exists."""
        try:
            self.minioClient.make_bucket(self.bucket_name)
        except S3Error as err:
            if err.code == 'BucketAlreadyOwnedByYou':
                logger.info("Bucket already exists.")
            else:
                logger.error(f"Error creating bucket: {str(err)}")
                raise

    def check_file_uploaded(self, file_name):
        """
        Check if a file exists in the Minio bucket.

        Parameters:
        file_name (str): The name of the file to check.

        Returns:
        bool: True if the file exists, False otherwise.
        """
        try:
            self.minioClient.stat_object(self.bucket_name, file_name)
            return True  # The file exists
        except S3Error as e:
            if e.code == "NoSuchKey":
                return False  # The file does not exist
            else:
                logger.error(f"Error checking file {file_name}: {str(e)}")
                raise  # Reraise the exception if it's not a missing file issue
