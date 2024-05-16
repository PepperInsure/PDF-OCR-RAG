This is not production ready. Considerations:
Alter defaults for minio in code and the minio deployment.
Change bucket name if needed.
Remove the --reload flag from the Dockerfile.

Make an .env file with these two lines below in the root directory:
OPENAI_API_KEY=[your-open-ai-api-key-here]
ALLOW_RESET=TRUE

Make sure docker is running.

To run both minio and main docker together:
docker-compose up --build

To start the main server without docker but load minio in docker:
docker network create mynetwork
docker run --network my-network --name festive-robinson -d -p 9000:9000 -p 9001:9001 minio/minio server /data --console-address ":9001"
uvicorn main:app --reload




Setup before dockerization:
python -m pip install fastapi uvicorn[standard]
pip install minio
python -m pip install chromadb
python -m pip install python-dotenv
python -m pip install langchain==0.1.0 openai==1.7.2 langchain-openai==0.0.2 langchain-community==0.0.12 langchainhub==0.1.14
pip install pytest
pip install pytest-asyncio

One collection for each doc.


Considerations:
GPT-4o is said to be better at non-english languages.
Should I use fixed size chunking or newline based?
Japanese specific preprocessing, embedding, vectorsearch, or langdetect if needed.

Make the tests work!