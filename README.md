# FlairLabs ChatBot

## Install Requirements

Run the following commands in order
```
npm install
pip install -r requirements.txt
```

## Project Structure

```
├───apps
│   ├───chat-be             <---- Express-Node backend app
│   ├───chat-fe             <---- Angular-Tailwind fronetend app
│   └───llm-rag             <---- FastAPI + LlamaIndex RAG server
├───demo-assets
├───libs
│   ├───backend
│   │   └───chatbot-core    <---- TRPC Server config
│   └───frontend
│       └───utils
│           └───api-client  <---- TRPC Client config
├───uploads                 <---- directory for all RAG file uploads
```

## Configuration for Rag Application

you can use any variation of these 2 JSONs

```JSON
{
    "vector_database": "KDB",
    "use_cohere_reranking": true,
    "embedding_model": "text-embedding-3-small",
    "chunking_size": 1536
}
```
```JSON
{
    "vector_database": "PINECONE",
    "use_cohere_reranking": false,
    "embedding_model": "voyage-2",
    "chunking_size": 1024
}
```

## Start the application

1. Run `npm run dev:be` to start the Node backend app
2. Run `npm run dev:fe` to start the Angular fronetend app
3. Run `npm run rag-server` to start the RAG server

## Explanation

1. The RAG Server consists of :
    - Parse the uploaded PDFs from the `uploads\` folder.
    - Clean the Embedding table in KDB.AI Vector Database and initialize the table
    - Configure a MarkdownElementNodeParser and recursive retrieval RAG technique to hierarchically index and query over tables and text in the uploaded documents
    - Get the RAG QueryEngine configured with Post processing Rerank embedding model (Cohere) which is used to RAG over complex documents that can answer questions over both tabular and unstructured data
    - A FastAPI server to trigger the file upload process and to query the RAG Server
2. The Angular Frontend app consists of:
    - A TRPC Client to communicate with the backend server
    - An Upload Component which accepts documents to be uploaded
    - A Chat component to interact with the AI assistant to fetch details from the previously uploaded PDFs
3. The Node backend app consists of:
    - `/api` route to connect the TRPC Server, which is used to communicate with the browser client
    - `/upload` route to accept files, to circumvent the issue of multipart-formdata not supported by trpc at the moment
4. The project uses Nx Monorepo framework to host all the above apps