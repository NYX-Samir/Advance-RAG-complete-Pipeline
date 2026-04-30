from typing import List, Tuple
from langchain_core.documents import Document

from src.chunking import Chunking
from src.context_compression import ContextCompressor
from src.documents_ingestion import DataIngestion
from src.generator_with_citations import RAGGenerator
from src.llm_client import GroqLLM
from src.query_transformer import QueryTransformer
from src.reranker import ReRanker
from src.retrieval import HybridRetriever
from src.vector_embedding import EmbeddingStore


class RAGPipeline:

    def __init__(
        self,
        data_paths: List[str],
        persist_dir: str = "./chroma_db",

        chunking_mode: str = "recursive",
        enable_rerank: bool = True,
        enable_compression: bool = True,
        top_k: int = 5,
        verbose: bool = True,
    ):
        self.data_paths = data_paths
        self.persist_dir = persist_dir

        self.chunking_mode = chunking_mode.lower()
        self.enable_rerank = enable_rerank
        self.enable_compression = enable_compression
        self.top_k = top_k
        self.verbose = verbose

        # Runtime objects
        self.chunks = None
        self.vector_db = None
        self.retriever = None
        self.reranker = None

        self.query_transformer = QueryTransformer()
        self.llm = None
        self.compressor = None
        self.generator = None

        if self.verbose:
            print("RAG pipeline config initialized.")

    # ------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------
    def build_index(self, rebuild: bool = False):

        if self.verbose:
            print("\n[INDEX] Building / Loading index...")
            print("Loading documents...")

        docs = DataIngestion.ingest(self.data_paths)

        if self.verbose:
            print(f"Chunking mode: {self.chunking_mode}")

        if self.chunking_mode == "recursive":
            self.chunks = Chunking.recursive_chunking(docs)

        elif self.chunking_mode == "semantic":
            self.chunks = Chunking.semantic_chunking(docs)

        else:
            raise ValueError(f"Invalid chunking mode: {self.chunking_mode}")

        if self.verbose:
            print(f"Generated {len(self.chunks)} chunks.")
            print("Creating / loading vector database...")

        embedding_store = EmbeddingStore()
        self.vector_db = embedding_store.create_or_load_db(
            chunks=self.chunks,
            persist_directory=self.persist_dir,
            rebuild=rebuild,
        )

        if self.verbose:
            print("Building hybrid retriever...")

        self.retriever = HybridRetriever(self.vector_db, self.chunks)

        if self.verbose:
            print("[INDEX] Ready.")

    def rebuild_index(self):
        if self.verbose:
            print("\n[INDEX] Rebuilding index...")
        self.build_index(rebuild=True)

    # ------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------
    def load_models(self):

        if self.verbose:
            print("\n[MODELS] Loading models...")

        if self.enable_rerank:
            self.reranker = ReRanker()
            if self.verbose:
                print("Re-ranker enabled.")
        else:
            self.reranker = None
            if self.verbose:
                print("Re-ranker disabled.")

        self.llm = GroqLLM()

        if self.enable_compression:
            self.compressor = ContextCompressor(self.llm)
            if self.verbose:
                print("Context compression enabled.")
        else:
            self.compressor = None
            if self.verbose:
                print("Context compression disabled.")

        self.generator = RAGGenerator()
        self.generator.llm=self.llm
 
        if self.verbose:
            print("[MODELS] Ready.")

    # ------------------------------------------------
    # Safety check
    # ------------------------------------------------
    def ready(self):

        if self.retriever is None or self.vector_db is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        if self.generator is None or self.llm is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        if self.verbose:
            print("Pipeline is ready for queries.")

    # ------------------------------------------------
    # Query execution
    # ------------------------------------------------
    def run(self, query: str) -> str:

        self.ready()

        if self.verbose:
            print(f"\nUser Query: {query}")

        # Query transformation
        queries = self.query_transformer.multi_query(query)

        if self.verbose:
            print(f"Expanded Queries: {queries}")

        # Retrieval
        retrieved: List[Tuple[Document, float]] = []

        for q in queries:
            retrieved.extend(
                self.retriever.retrieve(q, k=self.top_k)
            )

        if not retrieved:
            return "No relevant documents found."

        # Deduplicate
        unique_docs = {}
        for doc, score in retrieved:
            key = doc.page_content[:200]
            if key not in unique_docs:
                unique_docs[key] = (doc, score)

        retrieved_docs = [v[0] for v in unique_docs.values()]

        if self.verbose:
            print(f"Retrieved {len(retrieved_docs)} unique documents.")

        # Re-ranking
        if self.enable_rerank and self.reranker:
            reranked = self.reranker.rerank(
                query, retrieved_docs, top_n=self.top_k
            )
            reranked_docs = [doc for doc, _ in reranked]
            if self.verbose:
                print("Reranking applied.")
        else:
            reranked_docs = retrieved_docs[:self.top_k]
            if self.verbose:
                print("Reranking skipped.")

        # Context compression
        if self.enable_compression and self.compressor:
            compressed_docs = self.compressor.compress_documents(
                query=query,
                documents=reranked_docs,
            )
            if self.verbose:
                print("Compression applied.")
        else:
            compressed_docs = reranked_docs
            if self.verbose:
                print("Compression skipped.")

        # Generation
        answer = self.generator.generate_with_citations(
            query=query,
            context_docs=compressed_docs,
        )

        return answer

    
    def retrieve_for_evaluation(self, query: str, k: int):
    
        retrieved = self.retriever.retrieve(query, k=20)

        docs = [doc for doc, _ in retrieved]
        
        if self.enable_rerank and self.reranker:
            reranked = self.reranker.rerank(query, docs, top_n=k)
            return reranked

        return retrieved[:k]
