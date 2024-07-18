from fastapi import FastAPI
import uvicorn
from common_types import RagPipeline, QueryDocs
import pathlib
from rag_llama_index import get_rag_query_engine, get_answer_with_sources


app = FastAPI()
rag_pipeline = RagPipeline()

current_file_path = pathlib.Path(__file__).parent.resolve()

@app.get("/load_docs")
def load_docs() -> dict[str, str]:
    rag_pipeline.query_engine = get_rag_query_engine(current_file_path)
    return {"messsage": "success"}


@app.post("/answer")
async def get_answer(query_docs: QueryDocs):
    return get_answer_with_sources(rag_pipeline.query_engine, query_docs.query, current_file_path)


if __name__ == "__main__":
    uvicorn.run(app, port=4444)
