import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util

EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDER_NAME)

def load_vector_store(path="./data/processed/complaint_embeddings.parquet", sample_size=1000):
    df = pd.read_parquet(path)
    return df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)

def retrieve(question, vector_store, k=3):
    q_embedding = embedder.encode(question, convert_to_tensor=True)
    corpus_embeddings = [torch.tensor(e, dtype=torch.float32) for e in vector_store["embedding"]]
    hits = util.semantic_search(q_embedding, corpus_embeddings, top_k=k)[0] # type: ignore
    return [vector_store.iloc[hit["corpus_id"]]["document"] for hit in hits]
