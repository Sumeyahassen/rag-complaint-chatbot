import pandas as pd  # type: ignore
import torch  # type: ignore
from sentence_transformers import SentenceTransformer, util  # type: ignore
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODEL = "google/flan-t5-base"

_embedder = None
_tokenizer = None
_generator = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDER_NAME)
    return _embedder


def load_vector_store(path="./data/processed/complaint_embeddings.parquet", sample_size=1000):
    df = pd.read_parquet(path)
    n = min(sample_size, len(df))
    return df.sample(n=n, random_state=42).reset_index(drop=True)


def retrieve(question, vector_store, k=3):
    embedder = get_embedder()
    q_embedding = embedder.encode(question, convert_to_tensor=True)
    corpus_embeddings = [torch.tensor(e, dtype=torch.float32) for e in vector_store["embedding"]]
    hits = util.semantic_search(q_embedding, corpus_embeddings, top_k=k)[0] # type: ignore
    results = []
    for hit in hits:
        idx = hit["corpus_id"]
        results.append(vector_store.iloc[idx]["document"])
    return results


def build_prompt(context_chunks, question):
    context = "\n".join(context_chunks)
    return f"""You are a financial analyst assistant for CrediTrust.
Summarize the main issues in 2–3 sentences.
If the context does not contain the answer, say you do not have enough information.

Complaints:
{context}

Question: {question}
Answer:"""


def get_generator():
    global _tokenizer, _generator
    if _generator is None:
        _tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
        _generator = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL)
    return _tokenizer, _generator


def generate_answer(prompt):
    tokenizer, model = get_generator()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512) # type: ignore
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True) # type: ignore


# --- Evaluation Loop ---
if __name__ == "__main__":
    vector_store = load_vector_store(sample_size=1000)

    test_questions = [
        "What are the most common credit card complaints?",
        "What issues do consumers report about loans?",
        "Are there complaints about savings accounts?",
        "What problems occur with money transfers?",
    ]

    print("| Question | Answer |")
    print("|----------|--------|")

    for q in test_questions:
        chunks = retrieve(q, vector_store, k=3)
        prompt = build_prompt(chunks, q)
        answer = generate_answer(prompt)
        print(f"| {q} | {answer} |")
