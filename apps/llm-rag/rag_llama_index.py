from llama_parse import LlamaParse
from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
    StorageContext,
)
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.kdbai import KDBAIVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from common_types import Configuration
from pinecone import Pinecone, ServerlessSpec

import json
import kdbai_client as kdbai
import os
from common_constants import (
    OPENAI_API_KEY,
    COHERE_API_KEY,
    LLAMA_CLOUD_API_KEY,
    KDBAI_API_KEY,
    KDBAI_ENDPOINT,
    KDBAI_TABLE_NAME,
    GENERATION_MODEL,
    OPEN_AI_EMBEDDING_MODEL,
    VOYAGER_API_KEY,
    VOYAGER_EMBEDDING_MODEL,
    KDB_VECTOR_STORE,
    PINECONE_VECTOR_STORE,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)
from typing import List

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["COHERE_API_KEY"] = COHERE_API_KEY
os.environ["LLAMA_CLOUD_API_KEY"] = LLAMA_CLOUD_API_KEY


def get_config(current_file_path: str) -> Configuration:
    with open(f"{current_file_path}/config.json") as config_file:
        data = json.load(config_file)

    return data


def initialize_vector_store_and_storage_context(
    vector_db_name: str, vector_dimension: int
) -> StorageContext:
    if vector_db_name == KDB_VECTOR_STORE:
        session = kdbai.Session(api_key=KDBAI_API_KEY, endpoint=KDBAI_ENDPOINT)

        schema = dict(
            columns=[
                dict(name="document_id", pytype="bytes"),
                dict(name="text", pytype="bytes"),
                dict(
                    name="embedding",
                    vectorIndex=dict(type="flat", metric="L2", dims=vector_dimension),
                ),
            ]
        )

        # First ensure the table does not already exist
        if KDBAI_TABLE_NAME in session.list():
            session.table(KDBAI_TABLE_NAME).drop()

        # Create the table
        table = session.create_table(KDBAI_TABLE_NAME, schema)
        vector_store = KDBAIVectorStore(table)

    elif vector_db_name == PINECONE_VECTOR_STORE:
        pc = Pinecone(api_key=PINECONE_API_KEY)

        if PINECONE_INDEX_NAME in pc.list_indexes().names():
            pc.delete_index(PINECONE_INDEX_NAME)

        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=vector_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    return StorageContext.from_defaults(vector_store=vector_store)


def get_parsed_documents(current_file_path: str) -> List[Document]:
    upload_dir = str(os.path.join(current_file_path.parents[1], "uploads"))
    parsing_instructions = """This document contains many tables and graphs. Answer questions using the information in this article and be precise."""
    parser = LlamaParse(
        result_type="markdown", parsing_instructions=parsing_instructions
    )

    file_extractor = {".pdf": parser}
    return SimpleDirectoryReader(upload_dir, file_extractor=file_extractor).load_data()


def get_embedding_model(model: str):
    if model == OPEN_AI_EMBEDDING_MODEL:
        return OpenAIEmbedding(model=OPEN_AI_EMBEDDING_MODEL)
    else:
        return VoyageEmbedding(
            model_name=VOYAGER_EMBEDDING_MODEL, voyage_api_key=VOYAGER_API_KEY
        )


def get_rag_query_engine(current_file_path: str) -> BaseQueryEngine:
    config = get_config(current_file_path)

    llm = OpenAI(model=GENERATION_MODEL)

    Settings.llm = llm
    Settings.embed_model = get_embedding_model(config["embedding_model"])

    documents = get_parsed_documents(current_file_path)

    # Parse the documents using MarkdownElementNodeParser
    node_parser = MarkdownElementNodeParser(llm=llm, num_workers=8).from_defaults()

    # Retrieve nodes (text) and objects (table)
    nodes = node_parser.get_nodes_from_documents(documents)

    base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

    storage_context = initialize_vector_store_and_storage_context(
        config["vector_database"], config["chunking_size"]
    )

    # Create the index, inserts base_nodes and objects into KDB.AI
    recursive_index = VectorStoreIndex(
        nodes=base_nodes + objects, storage_context=storage_context
    )

    node_postprocessors = []
    if config["use_cohere_reranking"]:
        cohere_rerank = CohereRerank(top_n=10)
        node_postprocessors = [cohere_rerank]

    ### Create the query_engine to execute RAG pipeline using LlamaIndex, KDB.AI, and Cohere reranker
    query_engine = recursive_index.as_query_engine(
        similarity_top_k=15, node_postprocessors=node_postprocessors
    )

    return query_engine
