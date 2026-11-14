"""Evaluate retrieved evidence and preserve per-query traces."""
import re
from typing import Any


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _contains_evidence(chunk_text: str, evidence_quote: str) -> bool:
    c_norm = _norm(chunk_text)
    e_norm = _norm(evidence_quote)
    if not e_norm:
        return False
    if e_norm in c_norm:
        return True
    # Check 60-char core prefix/suffix or >= 75% content token overlap on the gold page
    if len(e_norm) >= 50 and (e_norm[:50] in c_norm or e_norm[-50:] in c_norm):
        return True
    e_tokens = [t for t in re.findall(r"[a-z0-9]+", e_norm) if len(t) > 3]
    if not e_tokens:
        return False
    matched = sum(1 for t in e_tokens if t in c_norm)
    return (matched / len(e_tokens)) >= 0.75


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Check source/page and enough gold evidence; keep hit@1 and hit@3 trace."""
    gold_source = question["gold_source_id"]
    gold_pages = question["gold_page"] if isinstance(question["gold_page"], list) else [int(question["gold_page"])]
    evidence_quote = question["evidence_quote"]
    req_filter = question.get("filter") or {}

    evaluated_hits = []
    for rank, h in enumerate(hits, start=1):
        source_match = (h.get("document_id") == gold_source) or (h.get("source") == gold_source)
        page_match = int(h.get("page", -1)) in gold_pages
        ev_match = _contains_evidence(h.get("text", ""), evidence_quote)
        is_hit = bool(source_match and page_match and ev_match)

        filter_ok = True
        for fk, fv in req_filter.items():
            actual_val = h.get(fk) or h.get("metadata", {}).get(fk)
            if str(actual_val) != str(fv):
                filter_ok = False

        hit_Copy = dict(h)
        hit_Copy.update({
            "rank": rank,
            "source_match": source_match,
            "page_match": page_match,
            "evidence_match": ev_match,
            "is_gold_hit": is_hit,
            "filter_compliant": filter_ok,
        })
        evaluated_hits.append(hit_Copy)

    hit_at_1 = 1 if (evaluated_hits and evaluated_hits[0]["is_gold_hit"]) else 0
    hit_at_3 = 1 if any(h["is_gold_hit"] for h in evaluated_hits[:3]) else 0

    # Check if evidence was split across multiple chunks on the gold page
    page_only_match = any(h["page_match"] for h in evaluated_hits[:3]) and not hit_at_3
    all_filter_compliant = all(h["filter_compliant"] for h in evaluated_hits[:3]) if req_filter else None

    return {
        "question_id": question["id"],
        "split": question["split"],
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "gold_source_id": gold_source,
        "gold_page": gold_pages[0] if len(gold_pages) == 1 else gold_pages,
        "evidence_quote": evidence_quote,
        "filter": req_filter if req_filter else None,
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "evidence_split_error": page_only_match,
        "filter_compliant_top3": all_filter_compliant,
        "retrieved_top3": evaluated_hits[:3],
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 by strategy and dev/test split."""
    summary: dict[str, Any] = {}
    strategies = sorted({t.get("strategy", "default") for t in traces})

    for strat in strategies:
        strat_traces = [t for t in traces if t.get("strategy", "default") == strat]
        summary[strat] = {}
        for split in ["dev", "test", "all"]:
            subset = strat_traces if split == "all" else [t for t in strat_traces if t["split"] == split]
            n = len(subset)
            if n == 0:
                continue
            h1 = sum(t["hit_at_1"] for t in subset)
            h3 = sum(t["hit_at_3"] for t in subset)
            split_errs = sum(1 for t in subset if t.get("evidence_split_error"))
            flt_qs = [t for t in subset if t.get("filter")]
            flt_ok = sum(1 for t in flt_qs if t.get("filter_compliant_top3") is True)

            summary[strat][split] = {
                "count": n,
                "hit_at_1_num": h1,
                "hit_at_1_str": f"{h1}/{n} ({h1 / n * 100:.1f}%)",
                "hit_at_1_rate": round(h1 / n, 4),
                "hit_at_3_num": h3,
                "hit_at_3_str": f"{h3}/{n} ({h3 / n * 100:.1f}%)",
                "hit_at_3_rate": round(h3 / n, 4),
                "evidence_split_errors": split_errs,
                "filter_questions_count": len(flt_qs),
                "filter_compliance_str": (
                    f"{flt_ok}/{len(flt_qs)} ({flt_ok / len(flt_qs) * 100:.1f}%)" if flt_qs else "N/A"
                ),
            }
    return summary
