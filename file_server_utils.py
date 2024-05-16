import os

from minio import Minio
from minio.error import S3Error
import io

# Load environment variables
minio_endpoint = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
minio_access_key = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
minio_secret_key = os.getenv('MINIO_SECRET_KEY', 'minioadmin')

class MinioFunctions:
    minioClient = None
    bucket_name = None
    def __init__(self, bucket_name):
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
        #TODO: Check if file is already uploaded
        try:
            file_content = await file.read()  # Read the file content into bytes
            file_stream = io.BytesIO(file_content)
            file_size = file.size  # Calculate the size of the file in bytes
            self.minioClient.put_object(
                self.bucket_name, file.filename, file_stream, file_size
            )
            file_uploaded = self.check_file_uploaded(file.filename)
            if file_uploaded:
                return {"message": "File uploaded successfully: " + file.filename}
            else:
                return {"message": "File upload failed"}
        except S3Error as err:
            return {"message": "Failed to upload file", "error": str(err)}

    def initialize_bucket(self):
        try:
            self.minioClient.make_bucket(self.bucket_name)
        except S3Error as err:
            if err.code == 'BucketAlreadyOwnedByYou':
                print("Bucket already exists.")
            else:
                raise

    def check_file_uploaded(self, file_name):
        try:
            stat = self.minioClient.stat_object(self.bucket_name, file_name)
            return True  # The file exists
        except S3Error as e:
            if e.code == "NoSuchKey":
                return False  # The file does not exist
            else:
                raise  # Reraise the exception if it's not a missing file issue
