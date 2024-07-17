from llama_index.core.base.base_query_engine import BaseQueryEngine
from pydantic import BaseModel

class QueryDocs(BaseModel):
    query: str


class RagPipeline:
    query_engine: BaseQueryEngine


class Configuration(BaseModel):
    vector_database: str
    use_cohere_reranking: bool
    embedding_model: str
    chunking_size: int