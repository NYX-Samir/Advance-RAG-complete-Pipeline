import time
import sys
import os
from statistics import mean

# Ensures that the src module can be found if running from terminal
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

sys.path.append(BASE_DIR)

from src.rag_pipeline import RAGPipeline
from src.evaluation_metrics import RAGEvaluator
from evaluation.eval_dataset import EVALUATION_DATASET

# --- Configuration ---
TOP_K = 5
INSPECTION_MODE = False
CONFIG_NAME = "Hybrid (No Compression)" # Table mein display karne ke liye

def run_evaluation():
    print(f"Starting Evaluation for Config: {CONFIG_NAME}...")
    
    # 1. Initialize Pipeline
    pipeline = RAGPipeline(
        data_paths=[DATA_DIR],
        persist_dir=DB_DIR,
        chunking_mode="recursive",
        enable_compression=False,
        enable_rerank=False,
        top_k=TOP_K,
        verbose=False,
    )
    
    # 2. Build Index & Load Models
    pipeline.build_index(rebuild=False)
    pipeline.load_models()
    
    # 3. Tracking Metrics
    recall_scores = []
    precision_scores = []
    mrr_scores = []
    latencies = []
    
    # 4. Evaluation Loop
    for idx, item in enumerate(EVALUATION_DATASET, 1):
        query = item["query"]
        relevant_uids = item["relevant_uids"]
        
        print(f"\n[{idx}/{len(EVALUATION_DATASET)}] Query: {query}")
        
        # Start Timer
        start_time = time.time()
        
        # Retrieve chunks (specifically for evaluation, skipping generator)
        retrieved = pipeline.retrieve_for_evaluation(query, k=TOP_K)
        
        # Stop Timer
        latency = time.time() - start_time
        
        # Inspection Mode (For debugging what chunks are retrieved)
        if INSPECTION_MODE:
            for doc, score in retrieved:
                uid = RAGEvaluator._doc_uid(doc)
                print(f" - UID: {uid} | Score: {score:.4f}")
                print(f"   Preview: {doc.page_content[:150]}...\n")
                
        # Calculate Metrics
        recall = RAGEvaluator.calculate_recall_at_k(
            retrieved_docs=retrieved,
            relevant_doc_uids=relevant_uids,
            k=TOP_K
        )
        
        precision = RAGEvaluator.calculate_precision_at_k(
            retrieved_docs=retrieved,
            relevant_doc_uids=relevant_uids,
            k=TOP_K
        )
        
        mrr = RAGEvaluator.calculate_mrr(
            retrieved_docs=retrieved,
            relevant_doc_uids=relevant_uids
        )
        
        # Store for aggregation
        recall_scores.append(recall)
        precision_scores.append(precision)
        mrr_scores.append(mrr)
        latencies.append(latency)
                
        print(f"   -> Recall@{TOP_K}: {recall:.3f} | Precision@{TOP_K}: {precision:.3f} | MRR: {mrr:.3f} | Latency: {latency:.2f}s")
        
    # 5. Print Aggregated Results in Final Table
    print("\n\n" + "="*85)
    print(" " * 25 + "LOCAL LLM RAG EVALUATION RESULTS")
    print("="*85)
    print(f"{'Config':<30} | {'Recall@'+str(TOP_K):<8} | {'Precision@'+str(TOP_K):<11} | {'MRR':<5} | {'Latency'}")
    print("-" * 85)
    
    avg_recall = mean(recall_scores)
    avg_precision = mean(precision_scores)
    avg_mrr = mean(mrr_scores)
    avg_latency = mean(latencies)
    
    print(f"{CONFIG_NAME:<30} | {avg_recall:<8.3f} | {avg_precision:<11.3f} | {avg_mrr:<5.3f} | {avg_latency:.2f}s")
    print("="*85 + "\n")

if __name__ == "__main__":
    run_evaluation()