"""
JanNyaya AI - Multi-Hop Legal RAG Engine

Responsibilities:
1. Execute multi-step legal research across multiple statutory disciplines.
2. For complex queries (e.g., debt default + limitation + civil recovery + NI Act):
   - Hop 1: Substantive legal relationship / obligation (Contract Act / NI Act / Property Act)
   - Hop 2: Procedural enforcement route (CPC / Summary Suit / Lok Adalat / Arbitration)
   - Hop 3: Limitation period & statutory prerequisites (Limitation Act 1963 / Notice periods)
3. Merge cross-hop evidence via Reciprocal Rank Fusion (RRF) and authority weighting.
4. Deduplicate across hops while preserving all unique statutory sections.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:
    from backend.retriever import hybrid_search, RRF_K
    from backend.legal_query_planner import QueryPlan
except ImportError:
    from retriever import hybrid_search, RRF_K
    from legal_query_planner import QueryPlan


class MultiHopLegalRAG:
    """Executes structured multi-hop legal retrieval."""

    @classmethod
    def execute_multi_hop_search(
        cls,
        plan: QueryPlan,
        semantic_k: int = 20,
        bm25_k: int = 20,
        final_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Performs retrieval for each sub-query in plan and fuses the results."""
        if not plan.sub_queries or len(plan.sub_queries) <= 1:
            return hybrid_search(
                plan.raw_query,
                semantic_k=semantic_k,
                bm25_k=bm25_k,
                final_k=final_k,
            )

        all_hop_results: List[List[Dict[str, Any]]] = []

        for sub_q in plan.sub_queries[:4]:
            try:
                hop_res = hybrid_search(
                    sub_q,
                    semantic_k=semantic_k,
                    bm25_k=bm25_k,
                    final_k=final_k,
                )
                if hop_res:
                    all_hop_results.append(hop_res)
            except Exception as e:
                print(f"[MultiHop] Sub-query error on '{sub_q}': {e}")

        if not all_hop_results:
            return hybrid_search(
                plan.raw_query,
                semantic_k=semantic_k,
                bm25_k=bm25_k,
                final_k=final_k,
            )

        # Merge results using Reciprocal Rank Fusion
        fused_scores: Dict[str, float] = {}
        doc_store: Dict[str, Dict[str, Any]] = {}

        for hop_index, hop_res in enumerate(all_hop_results):
            # Give slight priority to primary sub-query (hop 0)
            hop_weight = 1.2 if hop_index == 0 else 1.0

            for rank, item in enumerate(hop_res, start=1):
                if not isinstance(item, dict):
                    continue

                meta = item.get("metadata", {})
                sec = str(meta.get("section_number", meta.get("section", ""))).strip()
                title = str(meta.get("title", meta.get("source", ""))).strip()
                doc_key = f"{title}::Sec_{sec}::Chunk_{meta.get('chunk_index', 0)}"

                rrf_score = hop_weight / (RRF_K + rank)

                fused_scores[doc_key] = fused_scores.get(doc_key, 0.0) + rrf_score
                if doc_key not in doc_store:
                    doc_store[doc_key] = item

        # Sort merged documents by fused RRF score
        sorted_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)

        final_results: List[Dict[str, Any]] = []
        seen_sections = set()

        for k in sorted_keys:
            doc = doc_store[k]
            meta = doc.get("metadata", {})
            sec = str(meta.get("section_number", meta.get("section", ""))).strip()
            title = str(meta.get("title", meta.get("source", ""))).strip()
            sec_key = f"{title}:{sec}"

            # Ensure high diversity across distinct statutory sections
            if sec_key in seen_sections:
                continue
            seen_sections.add(sec_key)

            doc["multi_hop_score"] = round(fused_scores[k], 4)
            final_results.append(doc)

            if len(final_results) >= final_k:
                break

        return final_results


def execute_multi_hop_retrieval(
    plan: QueryPlan,
    semantic_k: int = 20,
    bm25_k: int = 20,
    final_k: int = 10,
) -> List[Dict[str, Any]]:
    """Helper to run multi-hop legal retrieval."""
    return MultiHopLegalRAG.execute_multi_hop_search(
        plan=plan,
        semantic_k=semantic_k,
        bm25_k=bm25_k,
        final_k=final_k,
    )
