# main.py
import json
import os
import shutil
from pprint import pprint

import dotenv
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pydantic import BaseModel, field_validator
from file_server_utils import MinioFunctions
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser

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
#OCR_SIM_DIRECTORY = Path(__file__).resolve().parent / "OCR_data"
OCR_SIM_DIRECTORY = Path("/app/OCR_data")
minioFunctions = MinioFunctions(BUCKET)

# chromaDB
#CHROMA_DIR = "./chroma_db"
#chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "/app/chroma_db"))
print(f"CHROMA_DIR: {CHROMA_DIR}")

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
print(f"ChromaDB client initialized with directory: {CHROMA_DIR}")

vector_db_outer = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=llm_embedding,
)

# Ensure directories exist
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(OCR_SIM_DIRECTORY, exist_ok=True)

# langchain
doc_creator = CharacterTextSplitter(separator='\n')

review_template_str = """Your task is to use the extracted content from OCR data to answer questions about a document.
Use the following context to answer the questions. Be as detailed as possible, but do not make up any information
that is not from the context. If you don't know an answer, say you don't know. Give one answer in english and one in japanese.
{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"], template=review_template_str
    )
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["question"], template="{question}")
)

messages = [review_system_prompt, review_human_prompt]

review_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"], messages=messages
)

# use the new gpt4-o said to be better for multiple languages
chat_model = ChatOpenAI(model="gpt-4-turbo", temperature=0)

output_parser = StrOutputParser()

print(f"Chroma vector store initialized with directory: {CHROMA_DIR}")

review_chain = (
        review_prompt_template
        | chat_model
        | output_parser
)


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
        collections = vector_db_outer._client.list_collections()
        all_collections = {}

        for collection in collections:
            collection_name = collection.name
            retriever = vector_db_outer.as_retriever(collection_name=collection_name)
            context_documents = retriever.get_relevant_documents(query="")
            all_collections[collection_name] = [doc.page_content for doc in context_documents]

        return {
            "message": "Retrieved documents from all collections.",
            "collections": all_collections
        }

    except Exception as e:
        return {"message": f"An error occurred: {str(e)}"}

@app.get("/list_documents")
async def list_documents(filename: str):
    collection_name = filename.replace(" ", "")
    ans = vector_db_outer.get()
    """
    try:
        # Use vector_db_outer to access the specified collection
        retriever = vector_db_outer.as_retriever(collection_name=collection_name)
        context_documents = retriever.get_relevant_documents(query="")
    except ValueError:
        return {"message": f"Collection '{collection_name}' not found in the vector database."}

    documents_list = [doc.page_content for doc in context_documents]
    print("Documents in collection:")
    for i, doc in enumerate(documents_list):
        print(f"Document {i + 1}: {doc}")
    
    return {
        "message": f"Retrieved {len(documents_list)} documents from collection '{collection_name}'.",
        "documents": documents_list
    }
    """
    return ans

@app.post("/reset")
async def reset_db():
    vector_db_outer._client.reset()
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
        # character

        # Filter out any empty lines if necessary
        documents_filtered = [line.strip() for line in document_lines if line.strip()]

        ids = [f"{collection_name}_{index}" for index, _ in enumerate(documents_filtered)]

        if not documents_filtered:
            return {"message": "No documents to index."}

        documents = doc_creator.create_documents(texts=documents_filtered)

        # Ensure the collection exists or create it if it does not

        chroma_client.get_or_create_collection(collection_name)

        print("collection name is: " + collection_name)

        vector_db_outer.add_documents(
            documents=documents,
            ids=ids,
            collection_name=collection_name,
            persist_directory=str(CHROMA_DIR),
        )
        vector_db_outer.persist()

        """vector_db_outer.add_documents(
            documents=documents,
            embedding=llm_embedding,
            collection_name=collection_name,
            persist_directory=str(CHROMA_DIR),
            ids=ids
        )"""

        retriever = vector_db_outer.as_retriever(collection_name=collection_name)
        context_documents = retriever.get_relevant_documents(query="")
        print(f"Indexed {len(context_documents)} documents in collection '{collection_name}'")

        return {
            "message": f"Imported Simulated OCR data to Vector DB. Indexed {len(context_documents)} documents in collection '{collection_name}'."}
    else:
        return {"message": "File not found"}


@app.post("/extract")
async def query_rag(filename: str, query: str):
    collection_name = filename.replace(" ", "")
    try:
        # chroma_collection = db.get_collection(collection_name)
        chroma_collection = vector_db_outer.get(collection_name)
        print(chroma_collection)
        # print(db.list_collections())

    except ValueError:
        return {"message": "File has not been put into the vector database."}

    retriever = vector_db_outer.as_retriever(k=5, collection_name=collection_name)
    context_documents = retriever.get_relevant_documents(query)

    # Debug: Print the retrieved document fragments with more details
    for i, doc in enumerate(context_documents):
        print(f"Retrieved fragment {i + 1}:")
        print(doc.page_content)
        print("=" * 50)

    max_tokens = 5000
    current_tokens = 0
    context = []

    for doc in context_documents:
        tokens = len(doc.page_content.split())
        if current_tokens + tokens > max_tokens:
            break
        context.append(doc.page_content)
        current_tokens += tokens

    context = "\n".join(context)

    # Ensure both context and query are strings
    if not isinstance(context, str):
        context = str(context)
    if not isinstance(query, str):
        query = str(query)

    # Prepare the inputs for the review chain
    inputs = {
        "context": context,
        "question": query
    }

    print(f"Context: {context}")
    print(f"Question: {query}")

    response = review_chain.invoke(inputs)

    return {
        "message": "Query processed successfully.",
        "response": response
    }


