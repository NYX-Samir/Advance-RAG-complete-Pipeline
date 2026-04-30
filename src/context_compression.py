from langchain_core.documents import Document
from typing import List
import copy

from src.llm_client import GroqLLM


class ContextCompressor:
    """
    Compresses documents using a local LLM (Ollama).
    Extracts only the most relevant sentences for a query.
    """

    def __init__(self, llm: GroqLLM, max_chars: int = 1500):
        self.llm = llm
        self.max_chars = max_chars

    @staticmethod
    def _extract_text(text: str) -> str:
        if not text:
            return ""
        return str(text).strip()

    def compress_documents(
        self,
        query: str,
        documents: List[Document],
        max_docs: int = 5,
    ) -> List[Document]:

        if not query or not query.strip():
            return []

        if not documents:
            return []

        compressed: List[Document] = []

        for doc in documents[:max_docs]:
            text = doc.page_content.strip()
            if not text:
                continue

            prompt = f"""
Extract only the sentences that directly answer or are relevant to the question below.
If no sentence is relevant, output exactly: None.
Do not add explanations or extra text.

Question:
{query}

Document:
{text[:self.max_chars]}

Relevant sentences:
""".strip()

            try:
                response = self.llm.generate(prompt)
                relevant_text = self._extract_text(response)

                if not relevant_text:
                    continue

                clean = relevant_text.strip().lower()

                if clean.startswith("none"):
                    continue

                compressed.append(
                    Document(
                        page_content=relevant_text,
                        metadata=copy.deepcopy(doc.metadata),
                    )
                )

            except Exception:
                compressed.append(
                    Document(
                        page_content=text,
                        metadata=copy.deepcopy(doc.metadata),
                    )
                )

        if not compressed:
            return documents[:max_docs]

        return compressed
