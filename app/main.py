from fastapi import FastAPI,HTTPException
from app.schemas import QueryRequest,QueryResponse
from typing import Optional
from src.rag_pipeline import RAGPipeline
import time
from app.logger import get_logger
from dotenv import load_dotenv
import os

load_dotenv()

DATA_PATH=os.getenv("DATA_PATH","./data")
CHROMA_DIR=os.getenv("CHROMA_DIR","./chroma_db")
CHUNKING_MODE=os.getenv("CHUNKING_MODE","recursive")
TOP_K=int(os.getenv("TOP_K",5))

ENABLE_RERANK=os.getenv("ENABLE_RERANK","true").lower()=="true"
ENABLE_COMPRESSION=os.getenv("ENABLE_COMPRESSION","true").lower()=="true"

REBUILD_INDEX_ON_STARTUP=(
    os.getenv("REBUILD_INDEX_ON_STARTUP","false").lower()=="true"
)


app=FastAPI(
    title="Production RAG API",
    version="1.0.0"
)
logger=get_logger("rap-api")
# Global pipeline instance
pipeline:Optional[RAGPipeline]=None

@app.on_event("startup")
def startup_event():
    
    global pipeline
    logger.info("Starting RAG API")
    
    if not os.path.exists("./data"):
        os.makedirs("./data")
        logger.warning("Data directory created dynamically.")
        
    try:
        pipeline=RAGPipeline(
            data_paths=["./data"],
            persist_dir="./chroma_db",
            chunking_mode="recursive",
            enable_rerank=True,
            enable_compression=True,
            top_k=5,
            verbose=False     
        )


        start=time.time()
        pipeline.build_index(rebuild=REBUILD_INDEX_ON_STARTUP)
        logger.info(f"Index ready in {time.time() - start:.2f}s")
        
        start=time.time()
        pipeline.load_models()
        logger.info(f"Models loaded in {time.time() - start:.2f}s")
        
        logger.info("RAG API startup complete with Groq backend")
        
    except Exception as e:
        logger.info("Startup failed")
        pipeline=None
        raise RuntimeError(f"Startup failed: {e}")

        
@app.get("/health")
def health_check():
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized"
        )
    return {"status":"ok"}


@app.post("/query",response_model=QueryResponse)
def query_rag(request:QueryRequest):
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized"
        )
    logger.info(f"Query received: {request.query}")
        
    start=time.time()
    
    try :
        answer=pipeline.run(request.query)
        duration = time.time() -start
        
        logger.info(f"Query complete in {duration:.2f}s")
        
        return QueryResponse(answer=answer)

    
    except Exception as e:
        logger.info("Query failed")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
        
@app.post("/rebuild-index")
def rebuild_index():
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized"
        )
    logger.warning("Index rebuild triggered")
    
    start =time.time()


    try:
        pipeline.rebuild_index()
        logger.warning(f"Index rebuild in {time.time()-start:.2f}s")
        return {"status":"index rebuild"}
    
    
    except Exception as e:
        logger.exception("Index rebuild failed")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
        