from backend.retriever import search_documents

questions = [
    "What is murder?",
    "What is the punishment for murder?",
    "What is rape?",
    "What is the punishment for rape?",
    "What is theft?",
    "What is the punishment for theft?",
    "What is snatching?",
    "What is the punishment for snatching?",
    "What is cheating?",
    "What is the punishment for cheating?",
]

for question in questions:

    results = search_documents(question, 5)

    print("\n==============================")
    print("QUESTION:", question)

    if results:
        print("INTENT:", results[0].get("query_intent"))
    else:
        print("INTENT: None")

    for result in results:
        print(
            "\nChunk:",
            result["metadata"].get("chunk_index"),
            "\nScore:",
            round(result["score"], 3),
            "\nText:",
            result["text"][:300].replace("\n", " ")
        )