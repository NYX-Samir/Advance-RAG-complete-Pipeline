from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import re 
import copy


class Chunking:
    
    
    @staticmethod
    def recursive_chunking(
        documents:List[Document],
        chunk_size:int=1000,
        chunk_overlap:int=200
    )->List[Document]:
        
        splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n","\n",".","!",",","?"," ",""]
        )
        
        return splitter.split_documents(documents)
    
    

@staticmethod
def semantic_chunking(
    documents: List[Document],
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=1000,
    similarity_percentile=30
) -> List[Document]:

    model = SentenceTransformer(model_name)
    chunks = []

    for doc in documents:
        text = doc.page_content.strip()
        if not text:
            continue

        sentences = re.split(r"(?<=[.!?])\s+", text)

        if len(sentences) <= 2:
            chunks.append(doc)
            continue

        embeddings = model.encode(sentences, normalize_embeddings=True)
        similarities = np.sum(embeddings[:-1] * embeddings[1:], axis=1)

        threshold = np.percentile(similarities, similarity_percentile)

        current_chunk = []
        current_length = 0

        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            current_length += len(sentence)

            split_here = i < len(similarities) and similarities[i] < threshold
            size_exceed = current_length >= chunk_size

            if split_here or size_exceed:
                chunk_text = " ".join(current_chunk).strip()
                if chunk_text:
                    chunks.append(
                        Document(
                            page_content=chunk_text,
                            metadata=doc.metadata.copy()
                        )
                    )
                current_chunk = []
                current_length = 0

        
        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if chunk_text:
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata=doc.metadata.copy()
                    )
                )

    return chunks
