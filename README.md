# OCR and Attribute Extraction API

## Overview

This project implements a Python-based backend API using FastAPI. The API provides endpoints for file uploads, simulates Optical Character Recognition (OCR), and uses Retrieval-Augmented Generation (RAG) for attribute extraction. The project demonstrates how to handle file uploads, perform OCR simulations, and extract attributes using advanced language models.

## Endpoints
## API Endpoints

### Upload Files

#### `POST /upload`

**Description:**
Uploads one or more files to Minio and creates a collection in ChromaDB for each file.

**Request:**
- `files`: List of files to be uploaded.

**Response:**
- A list of responses for each file indicating success or error.

### OCR to Vector Database

#### `POST /ocr`

**Description:**
Processes an uploaded file's OCR data and creates a collection in ChromaDB.

**Request:**
- `filename`: The name of the file to be processed.

**Response:**
- A message indicating the success of the OCR data import to the vector database.

### Extract Query

#### `POST /extract`

**Description:**
Extracts information from the vector database based on a query.

**Request:**
- `filename`: The name of the file to be queried.
- `query`: The query to be used for extraction.

**Response:**
- A response containing the query result.

## Extra Endpoints
### List Collections

#### `GET /list_collections`

**Description:**
Lists all the collections in ChromaDB.

**Response:**
- A list of all collections.

### List Documents

#### `GET /list_documents`

**Description:**
Lists all documents in a specified collection.

**Request:**
- `filename`: The name of the file for which the documents are to be listed.

**Response:**
- A list of documents in the specified collection.

### Reset Database

#### `POST /reset`

**Description:**
Resets ChromaDB by removing all collections and documents.

**Response:**
- A message indicating that ChromaDB has been reset.

## Manually using the endpoints
The root directory of the project contains a postman collection that can be imported:
`ocr_api_calls.postman_collection.json`

You may also look at documentation on a running server using: `http://127.0.0.1:8000/docs#`

## Setup

### Prerequisites

- Docker
- Python 3.10+
- An OpenAI API key

### Environment Variables

1. Create a `.env` file in the root directory with the following content:

```plaintext
OPENAI_API_KEY=[your-openai-api-key-here]
ALLOW_RESET=TRUE
```
2. Make sure Docker is running.

3. To run both Minio and main Docker together:
```commandline
docker-compose up --build
```

Alternatively, to start Minio in Docker but run the main app locally:
```commandline
docker network create mynetwork
docker run --network my-network --name festive-robinson -d -p 9000:9000 -p 9001:9001 minio/minio server /data --console-address ":9001"
uvicorn main:app --reload
```

## Information

- Each uploaded file gets its own collection.
- The query is a chain with a template string that asks for both English and Japanese, and not to hallucinate.
- If you wish to change this, look at `template_str` in `langchain_utils.py`.

## Discussions

### File Uploads

**Food for thought: Handling file uploads?**

- Using asynchronous file handling can improve performance. Using async on the upload endpoint with FastAPI ensures this.
- Integrating with a more scalable cloud storage service like S3 can enhance robustness.
- For large files chunked uploads may be appropriate.

### Language of Text

**Food for thought: Will the language of text matter in attribute extraction?**

- The query language can impact the performance of extraction accuracy. 
- The LLM will try to translate, but there may be differences when using japanese or english.
- Using multilingual models can improve handling documents in multiple languages. 
However: An attempt was made to use multilingual sentence transformers like 
paraphrase-multilingual-MiniLM-L12-v2, but performance ended up being best with multi-qa-MiniLM-L6-cos-v1.
- GPT-4o is said to be better at non-english languages.
- Consider the case of using fixed size chunking vs newline based.



