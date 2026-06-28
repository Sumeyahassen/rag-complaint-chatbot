import streamlit as st # type: ignore
from src.retriever import load_vector_store, retrieve
from src.generator import build_prompt, generate_answer

st.title("Complaint Chatbot")

vector_store = load_vector_store()

question = st.text_input("Ask a question:")
if question:
    chunks = retrieve(question, vector_store, k=3)
    prompt = build_prompt(chunks, question)
    answer = generate_answer(prompt)
    st.write("**Answer:**", answer)
