from retriever import load_vector_store, retrieve
from generator import build_prompt, generate_answer

if __name__ == "__main__":
    vector_store = load_vector_store()
    print("Welcome to the Complaint Chatbot! Type 'exit' to quit.\n")
    while True:
        question = input("Ask a question: ")
        if question.lower() == "exit":
            break
        chunks = retrieve(question, vector_store, k=3)
        prompt = build_prompt(chunks, question)
        answer = generate_answer(prompt)
        print("\nAnswer:", answer, "\n")
