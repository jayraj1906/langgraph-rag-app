from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List
import validators
import uvicorn
from src.config.astraDB_config import AstraDBConnection
from contextlib import asynccontextmanager
from src.components.split_text import SplitTexts
from src.logger import logging
from src.exception import MyException
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_groq import ChatGroq
import sys
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
import os
from dotenv import load_dotenv
from src.components.langgraph_workflow import WorkFlow
from fastapi import Request
from fastapi.responses import JSONResponse
from src.config.config import GROQ_API_KEY

db_conn = AstraDBConnection()

@asynccontextmanager
async def startup_event(app: FastAPI):
    # Load the ML model
    db_conn.initiate()
    app.state.vectorstore = db_conn.get_vector_store()
    astra_vector_index = VectorStoreIndexWrapper(vectorstore=db_conn.get_vector_store())
    app.state.retriever = app.state.vectorstore.as_retriever()
    api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=500)
    app.state.wiki=WikipediaQueryRun(api_wrapper=api_wrapper)
    app.state.llm=ChatGroq(groq_api_key=GROQ_API_KEY,model_name="Gemma2-9b-It")
    workflow = WorkFlow()
    workflow.initiate_workflow(app)
    app.state.workflow = workflow
    yield


app = FastAPI(lifespan=startup_event)
# In-memory storage for valid URLs
url_database = []

# Pydantic model for ingestion
class URLList(BaseModel):
    urls: List[str]

def is_valid_url(url: str) -> bool:
    return validators.url(url)


# Route 1: Ingestion of URLs
@app.post("/ingest")
async def ingest_urls(data: URLList):
    print(data)
    valid_urls = [url for url in data.urls if is_valid_url(url)]
    
    if not valid_urls:
        raise HTTPException(status_code=400, detail="No valid URLs provided.")
    try:
        text_splitter=SplitTexts(valid_urls)
        doc_lists=text_splitter.split_text()
        logging.info("Getting vector store after splitting")
        astra_vector_store=db_conn.get_vector_store()
        logging.info("Got the vector store after splitting")
        logging.info("Initiating splitted documents addition to the vector store db")
        astra_vector_store.add_documents(doc_lists)
        logging.info("Finished adding splitted documents into vector store")
        return "Inserted %i headlines." % len(doc_lists)
    except Exception as e:
        logging.error(f"Something went went wrong while storing the splitted text: {e}")
        raise MyException(e,sys)


# Dummy prediction route for demonstration
@app.post("/predict")
async def predict(request:Request,question: str):
    workflow: WorkFlow = request.app.state.workflow
    result = workflow.run(question)
    return result
    

@app.post("/predict-stream")
async def predict_stream(request: Request, question: str):
    workflow: WorkFlow = request.app.state.workflow
    result = workflow.run_streaming(question)
    print(result.get("documents")[0].page_content)
    print(type(result.get("documents")))
    # Optional: format output if you only want to return certain fields
    return JSONResponse(content={
        "question": result.get("question"),
        "final_description": result.get("documents")[0].page_content,
        "document_count": len(result.get("documents", [])),
    })

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)