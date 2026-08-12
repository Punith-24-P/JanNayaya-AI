from backend.rag_service import answer_question


questions = [
    "What is theft?",
    "What is the punishment for theft?",
    "What is murder?",
    "What is the punishment for murder?",
    "What is snatching?",
    "What is the punishment for snatching?",
    "What is rape?",
    "What is the punishment for rape?",
    "What is cheating?",
    "What is the punishment for cheating?",
]


for question in questions:

    print("\n" + "=" * 80)
    print("QUESTION:", question)
    print("=" * 80)

    try:
        result = answer_question(question, 5)

        print(result["answer"])

    except Exception as e:

        print("ERROR:", e)