from backend.retriever import hybrid_search, print_results


question = input(
    "Enter legal question: "
).strip()


if not question:

    print("Question cannot be empty.")

    raise SystemExit(1)


results = hybrid_search(
    question,
    semantic_k=10,
    bm25_k=10,
    final_k=5
)


print_results(results)