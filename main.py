# main.py
from langchain_utils import setup_chain

import json
import os

import dotenv
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, field_validator
from file_server_utils import MinioFunctions
from langchain.text_splitter import CharacterTextSplitter

from chromadb.utils import embedding_functions
import chromadb


class FileUpload(BaseModel):
    filename: str
    """
    TODO: Tests, error handling?
    """

    @field_validator('filename')
    def check_file_extension(cls, filename):
        print(filename)
        allowed_extensions = {'pdf', 'tiff', 'png', 'jpeg'}
        if filename.split('.')[-1].lower() not in allowed_extensions:
            raise ValueError('Unsupported file type')
        return filename


def change_extension_to_json(filename):
    # Create a Path object
    file = Path(filename)
    # Return with a new extension
    return file.with_suffix('.json')


# openAI APi key
dotenv.load_dotenv()
llm_embedding = OpenAIEmbeddings()

# fastAPI
app = FastAPI()
# Minio
BUCKET = "yourbucket"
# OCR_SIM_DIRECTORY = Path(__file__).resolve().parent / "OCR_data"
OCR_SIM_DIRECTORY = Path("/app/OCR_data")
minioFunctions = MinioFunctions(BUCKET)

# chromaDB
# CHROMA_DIR = "./chroma_db"
# chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "/app/chroma_db"))
print(f"CHROMA_DIR: {CHROMA_DIR}")

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
print(f"ChromaDB client initialized with directory: {CHROMA_DIR}")

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="multi-qa-MiniLM-L6-cos-v1"
)

# Ensure directories exist
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(OCR_SIM_DIRECTORY, exist_ok=True)

# langchain
doc_creator = CharacterTextSplitter(separator='\n')

review_chain = setup_chain()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/upload")
async def create_upload_file(file: UploadFile = File(...)):
    """
    TODO: Handle multiple files, checking if a file already exists, clearing out files for testing purposes, and invalid formats

    :param file:
    :return:
    """
    FileUpload(filename=file.filename)

    return await minioFunctions.upload_file(file)


@app.get("/list_collections")
async def list_collections():
    try:
        collections = chroma_client.list_collections()
        return collections
    except Exception as e:
        return {"message": f"An error occurred: {str(e)}"}


@app.get("/list_documents")
async def list_documents(filename: str):
    collection_name = filename.replace(" ", "")
    collection = chroma_client.get_collection(collection_name)
    documents = collection.get()
    return documents


@app.post("/reset")
async def reset_db():
    chroma_client.reset()
    return {"message": "ChromaDB has been reset, all collections and documents have been removed."}


@app.post("/ocr")
async def ocr_to_vector_db(filename: str):
    if minioFunctions.check_file_uploaded(filename):
        # file_path = OCR_SIM_DIRECTORY / change_extension_to_json(filename)
        file_path = Path(__file__).resolve().parent / "OCR_data" / change_extension_to_json(filename)
        print(file_path)
        if not file_path.is_file():
            # TODO: Should this give an HTTP code back?
            return {"message": "File is found but not part of OCR simulation."}

        collection_name = filename.replace(" ", "")

        with open(file_path, encoding='utf8') as file:
            data = json.load(file)

        document_lines = data['analyzeResult']['content'].split('\n')  # This splits the document at each newline

        # Filter out any empty lines if necessary
        documents_filtered = [line.strip() for line in document_lines if line.strip()]

        ids = [f"{collection_name}_{index}" for index, _ in enumerate(documents_filtered)]
        metadatas = [{"filename": filename} for index, _ in enumerate(documents_filtered)]

        if not documents_filtered:
            return {"message": "No documents to index."}

        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=embedding_func,
        )

        collection.add(
            ids=ids,
            documents=documents_filtered,
            metadatas=metadatas,
        )

        return {"message": "Imported Simulated OCR data to Vector DB."}
    else:
        return {"message": "File not found"}


@app.post("/extract")
async def query_rag(filename: str, query: str):
    collection_name = filename.replace(" ", "")
    try:
        chroma_collection = chroma_client.get_collection(collection_name)
        print(chroma_collection)

    except ValueError:
        return {"message": "File has not been put into the vector database."}
    context_docs = chroma_collection.query(
        query_texts=query,
        n_results=5,
    )
    print(context_docs)

    # Prepare the inputs for the review chain
    inputs = {
        "context": context_docs,
        "question": query
    }

    response = review_chain.invoke(inputs)

    return {
        "message": "Query processed successfully.",
        "response": response
    }
