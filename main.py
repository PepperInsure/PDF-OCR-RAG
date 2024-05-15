# main.py
import json
import os
import shutil
from pprint import pprint

import dotenv
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from langchain_community.vectorstores import Chroma
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


@app.get("/list_documents")
async def list_documents(filename: str):
    try:
        # Reinitialize vector_db_outer with the specific collection
        collections = vector_db_outer.get()
        return collections

    except Exception as e:
        return {"message": f"An error occurred: {str(e)}"}

@app.post("/reset")
async def reset_db():
    # Clear the ChromaDB directory to reset the database
    if CHROMA_DIR.exists() and CHROMA_DIR.is_dir():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # Reinitialize the vector_db_outer to ensure it's in sync with the cleared directory
    global vector_db_outer
    vector_db_outer = Chroma(
        chroma_client=chroma_client,
        persist_directory=str(CHROMA_DIR),
        embedding_function=llm_embedding,
    )
    print(f"ChromaDB has been reset, all collections and documents have been removed.")
    return {"message": "ChromaDB has been reset, all collections and documents have been removed."}

@app.post("/ocr")
async def ocr_to_vector_db(filename: str):
    if minioFunctions.check_file_uploaded(filename):
        file_path = OCR_SIM_DIRECTORY / change_extension_to_json(filename)
        print(file_path)
        if not file_path.is_file():
            # TODO: Should this give an HTTP code back?
            return {"message": "File is found but not part of OCR simulation."}

        collection_name = filename.replace(" ", "")
        """
        Do we want to prevent a repeat of the document from being put in?
        try:
            db.get_collection(collection_name)
            return {"message": "File has already been put into the vector database."}
        except ValueError:
            loader = JSONLoader(
            file_path=file_path,
            jq_schema='.analyzeResult.content',
            text_content=True)
        document = loader.load()
            
        """

        with open(file_path, encoding='utf8') as file:
            data = json.load(file)

        document = data['analyzeResult']['content']

        document_lines = data['analyzeResult']['content'].split(
            '\n')  # This splits the document at each newline character

        # Filter out any empty lines if necessary
        documents_filtered = [line.strip() for line in document_lines if line.strip()]

        ids = [f"{collection_name}_{index}" for index, _ in enumerate(documents_filtered)]

        if not documents_filtered:
            return {"message": "No documents to index."}

        documents = doc_creator.create_documents(texts=documents_filtered)

        Chroma.from_documents(
            documents=documents,
            embedding=OpenAIEmbeddings(),
            collection_name=collection_name,
            persist_directory=str(CHROMA_DIR),
            ids=ids
        )

        """vector_db_outer = Chroma(
            chroma_client=chroma_client,
            collection_name=collection_name,
            persist_directory=str(CHROMA_DIR),
            embedding_function=llm_embedding,
        )

        vector_db_outer.add_documents(
            documents=documents,
            embedding=OpenAIEmbeddings(),
            collection_name=collection_name,
            persist_directory=str(CHROMA_DIR),
            ids=ids
        )
        vector_db_outer.persist()"""


        # Verify if the collection now contains documents
        #print(vector_db_outer.get())

        """retriever = vector_db_outer.as_retriever(collection_name=collection_name)
        context_documents = retriever.get_relevant_documents(query="What is this document")
        print(f"Indexed {len(context_documents)} documents in collection '{collection_name}'")"""

        """
        vector_db = Chroma.from_documents(
            document, llm, collection_name=collection_name, ids=[collection_name],
            persist_directory="./chroma_db"
        )
        vector_db.persist()
        """

        return {"message": "Imported Simulated OCR data to Vector DB."}
    else:
        return {"message": "File not found"}


@app.post("/extract")
async def query_rag(filename: str, query: str):
    collection_name = filename.replace(" ", "")
    try:
        # chroma_collection = db.get_collection(collection_name)
        chroma_collection = vector_db_outer.get(collection_name)
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

    # print(f"Context: {context}")
    # print(f"Question: {query}")

    response = "Don't actually invoke until I figure this out"  # review_chain.invoke(inputs)

    """
    query_embedding = llm_embedding.embed_query(query)
    print(chroma_collection.query(query_embedding))

    vector_store = chroma.ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )

    query_engine = index.as_query_engine()
    response = query_engine.query("Give one answer in english, and one in japanese. What is this document about?")
    print(response)
    """

    return {
        "message": "Query processed successfully.",
        "response": response
    }


