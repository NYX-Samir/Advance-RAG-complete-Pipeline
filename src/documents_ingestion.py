#Libraries
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, DirectoryLoader,
    UnstructuredHTMLLoader, CSVLoader
)
from langchain_core.documents import Document
from typing import List
from pathlib import Path
import re


class DataIngestion:
    
    @staticmethod
    def load_pdf(file_path:str):
        return PyPDFLoader(file_path).load()
    
    @staticmethod
    def load_text(file_path:str):
        return TextLoader(file_path).load()
    
    @staticmethod
    def load_html(file_path:str):
        return UnstructuredHTMLLoader(file_path).load()
    
    @staticmethod
    def load_docx(file_path:str):
        return Docx2txtLoader(file_path).load()
    
    @staticmethod
    def load_csv(file_path:str):
        return CSVLoader(file_path).load()
    
    
    @staticmethod
    def load_directory(
        directory_path:str,
        glob_pattern:str="**/*" 
    ) -> List[Document]:
        loader=DirectoryLoader(
            directory_path,
            glob_pattern,
            show_progress=True,
        )
        return loader.load()
    
    @staticmethod
    def preprocess_text(text:str)->str:
        if not text:
            return ""
        
        text=re.sub(r"\s"," ",text)
        
        return text.strip()
     
    
    @staticmethod
    def normalize_metadata(doc:Document)->Document:
        source=(
            doc.metadata.get("source") or
            doc.metadata.get("file_path") or
            doc.metadata.get("path") or
            "unknown"
        )
        
        path = Path(source)
        
        doc.metadata = {
            "source":str(source),
            "filename":Path(source).name if source not in ("unknown","Unknown") 
            else "unknown",
            "page":doc.metadata.get("page","na"),
            "domain":path.parent.name.lower()
        }

        return doc
    
    

    @classmethod
    def ingest(cls, paths: List[str]) -> List[Document]:
        all_docs: List[Document] = []

        for base_path in paths:
            base = Path(base_path)

            files = (
                base.rglob("*") if base.is_dir() else [base]
            )

            for file_path in files:
                try:
                    if file_path.is_dir():
                        continue

                    suffix = file_path.suffix.lower()

                    if suffix == ".pdf":
                        docs = cls.load_pdf(str(file_path))
                    elif suffix == ".txt":
                        docs = cls.load_text(str(file_path))
                    elif suffix in [".html", ".htm"]:
                        docs = cls.load_html(str(file_path))
                    elif suffix == ".csv":
                        docs = cls.load_csv(str(file_path))
                    elif suffix == ".docx":
                        docs = cls.load_docx(str(file_path))
                    else:
                        continue

                    for doc in docs:
                        doc.page_content = cls.preprocess_text(doc.page_content)
                        doc = cls.normalize_metadata(doc)
                        all_docs.append(doc)

                except Exception as e:
                    print(f"[SKIP] {file_path} → {e}")

        print(f"[INGEST] Loaded documents: {len(all_docs)}")

        if not all_docs:
            raise RuntimeError("No documents loaded. Check PDF dependencies.")

        return all_docs


