import os
import torch
import uvicorn
import logging
import json
import shutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    PromptTemplate,
    Settings,
    load_index_from_storage,
    StorageContext,
)
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.ollama import Ollama
from langchain_huggingface import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Paths
docs_path = r"C:\Users\anubh\Documents\AI_Agent_LinkedIn_JobSearch\all_data\docs"
PERSIST_DIR = r"C:\Users\anubh\Documents\AI_Agent_LinkedIn_JobSearch\all_data\persistent_storage"
vector_index_dir = os.path.join(PERSIST_DIR, "vector")

# Check GPU availability
if torch.cuda.is_available():
    logger.info(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    logger.info("CUDA is not available. Using CPU.")

# Ollama Model Setup (Phi-3)
system_prompt = "Answer questions about a candidate's resume and profile concisely."
phi3_llm = Ollama(
    model="phi3:latest",
    temperature=0.01,
    request_timeout=400,
    system_prompt=system_prompt,
    context_window=2000
)
phi3_embed = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Prompt Template
qa_prompt_template_phi3 = """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge,
Answer the query in the format requested in the query.
Keep answers extremely short and concise (1-2 words where possible).
Do not elaborate unless necessary.
If options are provided, select one or more as required.
Return one answer unless specified otherwise.
For years of experience queries, return an integer > 1.
For "how many years of experience with [tool]", return an integer.
For "do you have experience with", return "Yes".
For "Experience with", return years as an integer.
For relocation or local queries, return "Yes".
Do not add extra text beyond the answer.

Query: {query_str}
Answer: \
"""

# Helper: Check if index exists and is valid
def check_persist(storage_path):
    if not os.path.exists(storage_path):
        logger.debug(f"No index directory found at {storage_path}")
        return False
    docstore_path = os.path.join(storage_path, "docstore.json")
    if not os.path.exists(docstore_path) or os.path.getsize(docstore_path) == 0:
        logger.debug(f"docstore.json missing or empty at {docstore_path}")
        return False
    try:
        with open(docstore_path, 'r') as f:
            json.load(f)
        logger.debug(f"Valid index found at {storage_path}")
        return True
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in docstore.json at {storage_path}")
        return False

# Build Vector Tool
def get_vector_tool(docs):
    if not docs:
        logger.error("No documents provided for indexing")
        raise ValueError("No documents to index")
    
    Settings.llm = phi3_llm
    Settings.embed_model = phi3_embed

    logger.debug(f"Creating/loading index in {vector_index_dir}")
    if check_persist(vector_index_dir):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=vector_index_dir)
            vector_index = load_index_from_storage(storage_context)
            logger.debug(f"Loaded index from {vector_index_dir}")
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}. Recreating index.")
            shutil.rmtree(vector_index_dir, ignore_errors=True)
            vector_index = VectorStoreIndex(docs)
            vector_index.storage_context.persist(vector_index_dir)
            logger.debug(f"Created and persisted new index to {vector_index_dir}")
    else:
        vector_index = VectorStoreIndex(docs)
        vector_index.storage_context.persist(vector_index_dir)
        logger.debug(f"Persisted new index to {vector_index_dir}")

    vector_query_engine = vector_index.as_query_engine(response_mode='compact', use_async=True)
    qa_prompt = PromptTemplate(qa_prompt_template_phi3)
    vector_query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt})

    vector_tool = QueryEngineTool.from_defaults(
        name="vector_tool",
        query_engine=vector_query_engine,
        description="Answer short and concise questions about the resume/profile."
    )
    logger.debug(f"Vector tool created with metadata: {vector_tool.metadata}")
    return vector_tool

# Load Documents
try:
    docs = SimpleDirectoryReader(docs_path).load_data()
    logger.debug(f"Loaded {len(docs)} documents from {docs_path}: {[doc.metadata.get('file_name', 'Unknown') for doc in docs]}")
    if not docs:
        raise ValueError("No documents found in docs_path")
except Exception as e:
    logger.error(f"Failed to load documents: {str(e)}")
    raise

# Set Up Query Engine
try:
    vector_tool = get_vector_tool(docs)
    vector_query_engine = vector_tool.query_engine
    logger.debug(f"Vector query engine initialized with tool: {vector_tool.metadata.name}")
except Exception as e:
    logger.error(f"Failed to initialize query engine: {str(e)}")
    raise

# FastAPI Setup
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

# POST Endpoint
@app.post("/resume_qa", response_model=QueryResponse)
async def generate_text_post(request: QueryRequest):
    try:
        logger.debug(f"Query received: {request.query}")
        response = vector_query_engine.query(request.query)
        if not response or not hasattr(response, 'response'):
            logger.error("Empty or invalid response from query engine")
            return QueryResponse(response="{}")
        logger.debug(f"Response content: {response.response}")
        return QueryResponse(response=str(response.response))
    except Exception as e:
        logger.error(f"Error in /resume_qa: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Health Check
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Run FastAPI Server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)