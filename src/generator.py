from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

GENERATOR_MODEL = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
generator = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL)

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

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = generator.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
