import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODELS = ["google/flan-t5-small", "google/flan-t5-base"]  # compare both

def load_vector_store(path="./data/processed/complaint_embeddings.parquet", sample_size=1000):
    df = pd.read_parquet(path)
    return df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)

def retrieve(question, vector_store, embedder, k=3):
    q_embedding = embedder.encode(question, convert_to_tensor=True)
    corpus_embeddings = [torch.tensor(e, dtype=torch.float32) for e in vector_store["embedding"]]
    hits = util.semantic_search(q_embedding, corpus_embeddings, top_k=k)[0] # type: ignore
    return [vector_store.iloc[hit["corpus_id"]]["document"] for hit in hits]

def build_prompt(context_chunks, question):
    context = "\n".join(context_chunks)
    return f"""You are a financial analyst assistant for CrediTrust.
Summarize the main issues in 2–3 sentences.
Be specific to the product mentioned.
If the context does not contain the answer, say you do not have enough information.

Complaints:
{context}

Question: {question}
Answer:"""

def generate_answer(prompt, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    vector_store = load_vector_store()
    embedder = SentenceTransformer(EMBEDDER_NAME)

    test_questions = [
        "What are the most common credit card complaints?",
        "What issues do consumers report about loans?",
        "Are there complaints about savings accounts?",
        "What problems occur with money transfers?",
        "Do consumers report problems with customer service?",
        "Are there complaints about interest rates?",
    ]

    for model_name in GENERATOR_MODELS:
        print(f"\n### Results using {model_name}\n")
        print("| Question | Answer | Retriever k=3 | Retriever k=5 |")
        print("|----------|--------|---------------|---------------|")

        for q in test_questions:
            chunks_k3 = retrieve(q, vector_store, embedder, k=3)
            chunks_k5 = retrieve(q, vector_store, embedder, k=5)

            prompt_k3 = build_prompt(chunks_k3, q)
            prompt_k5 = build_prompt(chunks_k5, q)

            answer_k3 = generate_answer(prompt_k3, model_name)
            answer_k5 = generate_answer(prompt_k5, model_name)

            print(f"| {q} | {answer_k3} | {answer_k3} | {answer_k5} |")
