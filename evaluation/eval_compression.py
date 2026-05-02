import time
import sys
import os
from statistics import mean

# Paths Setup
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
sys.path.append(BASE_DIR)

from src.rag_pipeline import RAGPipeline
from evaluation.eval_dataset import EVALUATION_DATASET

def run_compression_eval():
    print("Initializing pipeline... (Silently)\n")
    
    pipeline = RAGPipeline(
        data_paths=[DATA_DIR],
        persist_dir=DB_DIR,
        chunking_mode="recursive",
        enable_compression=True,
        enable_rerank=True,
        top_k=5,
        verbose=False,
    )
    
    pipeline.build_index(rebuild=False)
    pipeline.load_models()

    total_orig_chars = 0
    total_comp_chars = 0
    gen_times_uncompressed = []
    gen_times_compressed = []

    print("Evaluating queries (this may take a minute)...\n")

    for idx, item in enumerate(EVALUATION_DATASET, 1):
        # Progress indicator that overwrites the same line
        print(f"Processing query [{idx}/{len(EVALUATION_DATASET)}]...", end="\r", flush=True)

        query = item["query"]
        retrieved = pipeline.retrieve_for_evaluation(query, k=5)
        original_docs = [doc for doc, _ in retrieved]

        if not original_docs:
            continue

        # Measure original
        orig_chars = sum(len(doc.page_content) for doc in original_docs)
        total_orig_chars += orig_chars

        # Compress
        compressed_docs = pipeline.compressor.compress_documents(query=query, documents=original_docs)
        comp_chars = sum(len(doc.page_content) for doc in compressed_docs)
        total_comp_chars += comp_chars

        # Gen Time (Uncompressed)
        t0 = time.time()
        pipeline.generator.generate_with_citations(query=query, context_docs=original_docs)
        gen_times_uncompressed.append(time.time() - t0)

        # Gen Time (Compressed)
        t1 = time.time()
        pipeline.generator.generate_with_citations(query=query, context_docs=compressed_docs)
        gen_times_compressed.append(time.time() - t1)

    print(" " * 50, end="\r")

    saved_pct = ((total_orig_chars - total_comp_chars) / total_orig_chars * 100) if total_orig_chars > 0 else 0
    tokens_saved = (total_orig_chars - total_comp_chars) // 4

    print("==================================================")
    print("         COMPRESSION EVALUATION REPORT")
    print("==================================================")
    print(f"Original Context Size : {total_orig_chars} chars")
    print(f"Compressed Context    : {total_comp_chars} chars")
    print(f"Tokens Saved          : ~{tokens_saved} tokens")
    print(f"Overall Cost Reduction: {saved_pct:.2f}%")
    print("-" * 50)
    print(f"Avg Gen Time (Full)   : {mean(gen_times_uncompressed):.2f}s")
    print(f"Avg Gen Time (Comp)   : {mean(gen_times_compressed):.2f}s")
    print("==================================================\n")

if __name__ == "__main__":
    run_compression_eval()