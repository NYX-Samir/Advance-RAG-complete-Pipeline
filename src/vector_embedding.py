from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Optional
import os
import shutil
import torch


class EmbeddingStore:

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name

        self.embeddings = HuggingFaceEmbeddings(
            model_name=f"sentence-transformers/{model_name}",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

        print(f"Embedding model loaded on: {device}")


    @staticmethod
    def validate_chunks(chunks: List[Document]) -> List[Document]:
        valid = []

        for doc in chunks:
            if not isinstance(doc.page_content, str):
                continue
            if not doc.page_content.strip():
                continue
            valid.append(doc)

        if len(valid) == 0:
            raise ValueError("No valid chunks found for embeddings")

        return valid


    def create_or_load_db(
        self,
        chunks: List[Document],
        persist_directory: str = "./chroma_db",
        collection_name: str = "rag_collection",
        rebuild: bool = False,
    ) -> Chroma:

        chunks = self.validate_chunks(chunks)
        db_exists = os.path.exists(persist_directory) and os.listdir(persist_directory)


        if rebuild and db_exists:
            print("Rebuilding vector database...")
            shutil.rmtree(persist_directory)
            db_exists = False


        if db_exists:
            print("Loading existing Chroma database...")
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
                collection_name=collection_name,
            )


        print(f"Creating new Chroma database with {len(chunks)} chunks...")
        BATCH_SIZE = 5000 
        vectordb = None
        
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            print(f"Inserting batch {i//BATCH_SIZE + 1} (Size: {len(batch)} chunks)...")
            
            if vectordb is None:
                vectordb = Chroma.from_documents(
                    documents=batch,
                    embedding=self.embeddings,
                    persist_directory=persist_directory,
                    collection_name=collection_name,
                )
            else:
                vectordb.add_documents(documents=batch)

        return vectordb
