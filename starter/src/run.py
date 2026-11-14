"""Main pipeline entry point for Day 06 Data Ingestion Engineering lab."""
import csv
import json
import os
import statistics
import time
from pathlib import Path
from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .store import PgVectorStore


def main() -> None:
    load_dotenv()
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:55433/day6_lab")
    model_name = (os.getenv("EMBEDDING_MODEL") or "").strip() or "sentence-transformers/all-MiniLM-L6-v2"

    pdf_path = Path("data/kimball_dwt_3ed.pdf")
    benchmark_path = Path("benchmark/questions.json")
    results_dir = Path("results")
    root_results_dir = Path("../results")
    artifacts_dir = Path("../artifacts")

    results_dir.mkdir(parents=True, exist_ok=True)
    root_results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("1. Loading and normalizing PDF corpus (Pages 37..95)...")
    pages = load_pdf(
        path=pdf_path,
        document_id="kimball_dwt_2013",
        source="The Data Warehouse Toolkit 3rd Ed (2013)",
        page_range=(37, 95),
        log_path=results_dir / "ingestion_log.json",
    )
    if len(pages) < 30:
        raise RuntimeError(f"Corpus must have >= 30 valid pages, got {len(pages)}")
    print(f"   Loaded {len(pages)} valid content pages (skipped divider/blank pages logged).")

    with open(benchmark_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"   Loaded {len(questions)} benchmark questions.")

    print(f"2. Initializing local embedding model: {model_name} ...")
    embedder = MyEmbedder(model_name=model_name)
    store = PgVectorStore(database_url=db_url, dimensions=embedder.dimensions)
    store.create_schema()

    # Build chunks for A, B, C
    chunks_a = [c for p in pages for c in fixed_size(p, size=500, overlap=50)]
    chunks_b = [c for p in pages for c in by_paragraph(p, max_size=800)]
    chunks_c = [c for p in pages for c in by_paragraph(p, max_size=800)]

    configs = [
        {
            "strategy": "A",
            "name": "Fixed-size (500 chars, overlap 50) + No Filter",
            "chunk_method": "fixed_size(size=500, overlap=50)",
            "use_filter": False,
            "chunks": chunks_a,
            "hypothesis": "Baseline fixed window; fast and uniform, but often splits sentences and loses cross-paragraph context.",
        },
        {
            "strategy": "B",
            "name": "Paragraph/Structure-aware (max 800 chars) + No Filter",
            "chunk_method": "by_paragraph(max_size=800)",
            "use_filter": False,
            "chunks": chunks_b,
            "hypothesis": "Preserving paragraph and sentence boundaries will improve chunk coherence and evidence completeness over A.",
        },
        {
            "strategy": "C",
            "name": "Paragraph/Structure-aware (max 800 chars) + Metadata Filter (chapter)",
            "chunk_method": "by_paragraph(max_size=800)",
            "use_filter": True,
            "chunks": chunks_c,
            "hypothesis": "Adding chapter metadata filter on top of B will eliminate cross-chapter false positives on scoped queries.",
        },
    ]

    strategy_stats = {}
    all_traces = []

    print("3. Embedding, storing vectors in pgvector, and verifying idempotent reload...")
    for cfg in configs:
        strat = cfg["strategy"]
        chks = cfg["chunks"]
        t0 = time.perf_counter()
        vecs = embedder.embed_documents([c.text for c in chks])
        store.upsert(strat, chks, vecs)
        ingest_sec = round(time.perf_counter() - t0, 3)

        count_before = store.count_chunks(strat)
        # Re-ingest same chunks to verify idempotency (no duplicates)
        store.upsert(strat, chks, vecs)
        count_after = store.count_chunks(strat)

        lengths = [len(c.text) for c in chks]
        total_chars = sum(lengths)
        strategy_stats[strat] = {
            "strategy": strat,
            "description": cfg["name"],
            "chunk_method": cfg["chunk_method"],
            "metadata_filter_enabled": cfg["use_filter"],
            "num_chunks": len(chks),
            "min_chars": min(lengths),
            "avg_chars": round(statistics.mean(lengths), 1),
            "median_chars": round(statistics.median(lengths), 1),
            "max_chars": max(lengths),
            "total_chars": total_chars,
            "approx_tokens": round(total_chars / 4),
            "ingest_time_sec": ingest_sec,
            "api_cost_usd": 0.0,
            "db_count_initial": count_before,
            "db_count_after_reload": count_after,
            "idempotent_verified": count_before == count_after == len(chks),
        }
        print(
            f"   Strategy {strat}: {len(chks)} chunks | avg {strategy_stats[strat]['avg_chars']} chars | "
            f"time {ingest_sec}s | idempotent: {count_before}->{count_after}"
        )

    print("4. Running Top-3 retrieval across all 20 benchmark questions...")
    for cfg in configs:
        strat = cfg["strategy"]
        use_flt = cfg["use_filter"]
        for q in questions:
            q_vec = embedder.embed_query(q["question"])
            active_filter = q.get("filter") if (use_flt and q.get("filter")) else None
            hits = store.search(strategy=strat, query_vector=q_vec, top_k=3, filters=active_filter)
            trace = evaluate_query(q, hits)
            trace["strategy"] = strat
            trace["applied_filter"] = active_filter
            all_traces.append(trace)

    metrics_summary = summarize(all_traces)

    # Save full outputs to both starter/results and ../results
    output_bundle = {
        "embedding_model": model_name,
        "dimensions": embedder.dimensions,
        "vector_store": "PostgreSQL 16 + pgvector 0.8.7 (cosine distance)",
        "strategy_stats": strategy_stats,
        "metrics_summary": metrics_summary,
    }

    for out_d in [results_dir, root_results_dir]:
        with open(out_d / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(output_bundle, f, indent=2, ensure_ascii=False)
        with open(out_d / "traces_top3.json", "w", encoding="utf-8") as f:
            json.dump(all_traces, f, indent=2, ensure_ascii=False)
        with open(results_dir / "ingestion_log.json", "r", encoding="utf-8") as src_log:
            log_data = json.load(src_log)
        with open(out_d / "ingestion_log.json", "w", encoding="utf-8") as dst_log:
            json.dump(log_data, dst_log, indent=2, ensure_ascii=False)

    # Write experiment_log.csv
    exp_csv_path = artifacts_dir / "experiment_log.csv"
    with open(exp_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "strategy", "description", "embedding_model", "vector_store", "chunk_method",
            "metadata_filter", "num_chunks", "avg_chunk_chars", "ingest_time_sec", "cost_usd",
            "dev_hit_at_1", "dev_hit_at_3", "test_hit_at_1", "test_hit_at_3", "hypothesis"
        ])
        for cfg in configs:
            s = cfg["strategy"]
            st = strategy_stats[s]
            ms = metrics_summary[s]
            writer.writerow([
                s, cfg["name"], model_name, "pgvector", cfg["chunk_method"],
                cfg["use_filter"], st["num_chunks"], st["avg_chars"], st["ingest_time_sec"], 0.0,
                ms["dev"]["hit_at_1_str"], ms["dev"]["hit_at_3_str"],
                ms["test"]["hit_at_1_str"], ms["test"]["hit_at_3_str"],
                cfg["hypothesis"],
            ])

    print("\n=== RETRIEVAL METRICS SUMMARY ===")
    for s in ["A", "B", "C"]:
        ms = metrics_summary[s]
        print(
            f"Strategy {s} | DEV Hit@1: {ms['dev']['hit_at_1_str']} | DEV Hit@3: {ms['dev']['hit_at_3_str']} | "
            f"TEST Hit@1: {ms['test']['hit_at_1_str']} | TEST Hit@3: {ms['test']['hit_at_3_str']} | "
            f"Filter Compliance (All): {ms['all']['filter_compliance_str']}"
        )


if __name__ == "__main__":
    main()
