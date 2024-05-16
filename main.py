# main.py
from typing import List
import json
import os
import logging
from pathlib import Path

import dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, field_validator

from langchain_utils import setup_chain
from file_server_utils import MinioFunctions
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from chromadb.utils import embedding_functions
import chromadb

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# openAI API key
dotenv.load_dotenv()
llm_embedding = OpenAIEmbeddings()

# fastAPI
app = FastAPI()
# Minio
BUCKET = "yourbucket"
OCR_SIM_DIRECTORY = Path(os.getenv("OCR_DIR", Path(__file__).resolve().parent / "OCR_data"))

minioFunctions = MinioFunctions(BUCKET)

# chromaDB
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "/app/chroma_db"))
logger.info(f"CHROMA_DIR: {CHROMA_DIR}")

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
logger.info(f"ChromaDB client initialized with directory: {CHROMA_DIR}")

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="multi-qa-MiniLM-L6-cos-v1"
)

# Ensure directories exist
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(OCR_SIM_DIRECTORY, exist_ok=True)

# langchain settings
doc_creator = CharacterTextSplitter(separator='\n')
review_chain = setup_chain()


class FileUpload(BaseModel):
    """Model for validating file uploads."""
    filename: str

    @field_validator('filename')
    def check_file_extension(cls, filename):
        logger.info(f"Validating file extension for: {filename}")
        allowed_extensions = {'pdf', 'tiff', 'png', 'jpeg'}
        if filename.split('.')[-1].lower() not in allowed_extensions:
            raise ValueError('Unsupported file type')
        return filename


class OCRRequest(BaseModel):
    """Model for OCR request payload."""
    filename: str


class ExtractRequest(BaseModel):
    """Model for extraction request payload."""
    filename: str
    query: str


def change_extension_to_json(filename):
    """
    Change the file extension to .json.

    Parameters:
    filename (str): The original filename.

    Returns:
    Path: The new filename with .json extension.
    """
    # Create a Path object
    file = Path(filename)
    # Return with a new extension
    return file.with_suffix('.json')


async def create_collection(file_path, filename):
    """
    Create or update a collection in ChromaDB with OCR data.

    Parameters:
    file_path (Path): The path to the OCR data file.
    filename (str): The original filename.

    Raises:
    HTTPException: If no documents are found to index.
    """
    collection_name = filename.replace(" ", "")
    with open(file_path, 'r', encoding='utf8') as file:
        data = json.load(file)
    document_lines = data['analyzeResult']['content'].split('\n')  # This splits the document at each newline
    documents_filtered = [line.strip() for line in document_lines if line.strip()]
    ids = [f"{collection_name}_{index}" for index, _ in enumerate(documents_filtered)]
    metadatas = [{"filename": filename} for index, _ in enumerate(documents_filtered)]
    if not documents_filtered:
        raise HTTPException(status_code=400, detail="No documents to index.")

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_func,
    )

    # Check if documents already exist in the collection
    existing_documents = collection.get(ids=ids)
    if existing_documents:
        collection.delete(ids=ids)  # Remove existing documents to avoid duplicates

    collection.add(
        ids=ids,
        documents=documents_filtered,
        metadatas=metadatas,
    )


@app.get("/")
async def root():
    """
    Root endpoint.

    Returns:
    dict: A welcome message.
    """
    return {"message": "Hello World"}


@app.post("/upload")
async def create_upload_file(files: List[UploadFile] = File(...)):
    """
    Endpoint to handle file uploads.

    Parameters:
    files (List[UploadFile]): List of files to upload.

    Returns:
    list: List of responses for each file.
    """
    logger.info(f"Received files: {[file.filename for file in files]}")
    responses = []
    for file in files:
        try:
            FileUpload(filename=file.filename)
            response = await minioFunctions.upload_file(file)
            responses.append({"filename": file.filename, "status": "success", "response": response})
        except ValueError as ve:
            logger.error(f"ValueError for file {file.filename}: {str(ve)}")
            responses.append({"filename": file.filename, "status": "error", "detail": str(ve)})
        except Exception as e:
            logger.error(f"Exception for file {file.filename}: {str(e)}")
            responses.append({"filename": file.filename, "status": "error", "detail": str(e)})

    return responses


@app.get("/list_collections")
async def list_collections():
    """
    Endpoint to list all collections in ChromaDB.

    Returns:
    list: List of collections.
    """
    try:
        collections = chroma_client.list_collections()
        return collections
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/list_documents")
async def list_documents(filename: str):
    """
    Endpoint to list all documents in a collection.

    Parameters:
    filename (str): The name of the file corresponding to the collection.

    Returns:
    list: List of documents in the collection.
    """
    collection_name = filename.replace(" ", "")
    try:
        collection = chroma_client.get_collection(collection_name)
        documents = collection.get()
        return documents
    except ValueError:
        logger.error(f"Collection not found for filename: {filename}")
        raise HTTPException(status_code=404, detail="Collection not found")
    except Exception as e:
        logger.error(f"Error listing documents for filename {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/reset")
async def reset_db():
    """
    Endpoint to reset the ChromaDB.

    Returns:
    dict: A message indicating the reset status.
    """
    try:
        chroma_client.reset()
        return {"message": "ChromaDB has been reset, all collections and documents have been removed."}
    except Exception as e:
        logger.error(f"Error resetting ChromaDB: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/ocr")
async def ocr_to_vector_db(request: OCRRequest):
    """
    Endpoint to import OCR data into the vector database.

    Parameters:
    request (OCRRequest): The OCR request payload.

    Returns:
    dict: A message indicating the import status.
    """
    filename = request.filename
    if not minioFunctions.check_file_uploaded(filename):
        logger.error(f"File not found: {filename}")
        raise HTTPException(status_code=404, detail="File not found")
    file_path = OCR_SIM_DIRECTORY / change_extension_to_json(filename)
    logger.info(f"Processing OCR file: {file_path}")
    if not file_path.is_file():
        logger.error(f"File is found but not part of OCR simulation: {file_path}")
        raise HTTPException(status_code=404, detail="File is found but not part of OCR simulation.")
    try:
        await create_collection(file_path, filename)
    except Exception as e:
        logger.error(f"Error importing OCR data for filename {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    return {"message": "Imported Simulated OCR data to Vector DB."}


@app.post("/extract")
async def query_rag(request: ExtractRequest):
    """
    Endpoint to extract information using a query from the vector database.

    Parameters:
    request (ExtractRequest): The extraction request payload.

    Returns:
    dict: A message and the response to the query.
    """
    filename = request.filename
    query = request.query
    collection_name = filename.replace(" ", "")
    try:
        chroma_collection = chroma_client.get_collection(collection_name)
    except ValueError:
        logger.error(f"File not in vector database: {filename}")
        raise HTTPException(status_code=404, detail="File has not been put into the vector database.")
    except Exception as e:
        logger.error(f"Error querying vector database for filename {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    try:
        context_docs = chroma_collection.query(
            query_texts=query,
            n_results=5,
        )

        inputs = {
            "context": context_docs,
            "question": query
        }
        response = review_chain.invoke(inputs)
        return {
            "message": "Query processed. Query: " + query,
            "response": response
        }
    except Exception as e:
        logger.error(f"Error processing query for filename {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
