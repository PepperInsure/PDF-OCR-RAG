This is not production ready. Considerations:
Alter defaults for minio in code and the minio deployment.
Change bucket name if needed.
Remove the --reload flag from the Dockerfile.

To start the main server without docker:
uvicorn main:app --reload
Go to file_server_utils and replace festive-robinson with localhost
Note that because of jq this won't work on windows without specfically getting jq without pip

For both minio and main docker:
Use docker and run:
docker run --network my-network --name festive-robinson -d -p 9000:9000 -p 9001:9001 minio/minio server /data --console-address ":9001"

docker build -t my-fastapi-app .
docker run --network my-network -p 4000:80 --env-file .\.env -v ${PWD}:/app my-fastapi-app


Setup before dockerization:
python -m pip install fastapi uvicorn[standard]
pip install minio
python -m pip install chromadb
python -m pip install python-dotenv
python -m pip install langchain==0.1.0 openai==1.7.2 langchain-openai==0.0.2 langchain-community==0.0.12 langchainhub==0.1.14

One collection for each doc.


Considerations:
GPT-4o is said to be better at non-english languages.