# rag-complaint-chatbot
# RAG Complaint Chatbot

A Retrieval‑Augmented Generation (RAG) chatbot built using the CFPB consumer complaint dataset.  
This project is organized into tasks to gradually build the pipeline.

---

## 📊 Task 1: Exploratory Data Analysis
- **[EDA](ca://s?q=Task_1_EDA_in_RAG)** performed on the complaint dataset.
- Counted complaints per product (Credit Card, Loan, Savings, Money Transfer).
- Measured narrative lengths (average, min, max).
- Identified missing narratives (empty complaint texts).
- Deliverable: `src/data/preprocess.py` with EDA functions and cleaned dataset.

---

## 🧹 Task 2: Data Preprocessing
- **[Data cleaning](ca://s?q=Task_2_Data_Preprocessing)**: lowercasing, removing special characters.
- Tokenized and chunked long complaints into smaller segments.
- Generated embeddings using **MiniLM (sentence-transformers/all-MiniLM-L6-v2)**.
- Saved embeddings into `data/processed/complaint_embeddings.parquet`.
- Deliverable: `src/data/preprocess.py` updated with preprocessing + embedding logic.

---

##  Task 3: RAG Pipeline
- **[Retriever](ca://s?q=Retriever_in_RAG)**: loads vector store and retrieves top‑k complaint chunks for a question.
- **[Prompt template](ca://s?q=Prompt_engineering_in_RAG)**: guides the LLM to answer using retrieved context only.
- **[Generator](ca://s?q=LLM_for_customer_feedback)**: uses lightweight model (`flan‑t5-base` or `flan‑t5-small`) to produce answers.
- **[Evaluation](ca://s?q=Evaluation_in_AI_projects)**: tested with sample questions (e.g., “What are the most common credit card complaints?”).


