from typing import List
from langchain_core.documents import Document

from src.llm_client import GroqLLM


class RAGGenerator:
    """
    Final answer generator using local LLM (Ollama).
    Preserves the same behavior as the HuggingFace-based generator.
    """

    def __init__(
        self,
        max_content_chars: int = 6000,
    ):
        print("Initializing Local RAG Generator (Ollama)")
        self.llm = GroqLLM()
        self.max_content_chars = max_content_chars
        print("RAG Generator Ready")

    def generate_with_citations(
        self,
        query: str,
        context_docs: List[Document],
    ) -> str:

        if not query or not query.strip():
            return "Invalid query."

        if not context_docs:
            return "I cannot find this information in the provided source."


        context_text = ""
        total_chars = 0

        for i, doc in enumerate(context_docs):
            source = doc.metadata.get("source", "Unknown")
            block = f"\n[Source {i+1}]: {source}\n{doc.page_content}\n"

            total_chars += len(block)
            if total_chars > self.max_content_chars:
                break

            context_text += block


        prompt = f"""
Answer the question using ONLY the information in the context below.
Cite each factual statement using the exact source number where that fact appears. Do not combine sources unless both explicitly mention the same fact.
If the answer is not present in the context, say exactly:
"I cannot find this information in the provided source."

Context:
{context_text}

Question:
{query}

Answer with citations:
""".strip()


        try:
            answer = self.llm.generate(prompt)

            if not answer:
                return "I cannot find this information in the provided source."

            return answer.strip()

        except Exception as e:
            return f"Generation failed: {str(e)}"
