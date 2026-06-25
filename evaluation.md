# Task‑3 Evaluation

| Question | Answer | Quality (1–5) | Notes |
|----------|--------|---------------|-------|
| What are the most common credit card complaints? | late fees and damaging credit reports | 4 | Relevant, concise |
| What issues do consumers report about loans? | late fees and damaging credit reports | 2 | Too generic, not loan‑specific |
| Are there complaints about savings accounts? | Yes | 1 | Too vague |
| What problems occur with money transfers? | Detailed narrative | 3 | Informative but too long |

---

## Notes on Adjustments
- **Prompt template**: Needs clearer instructions (e.g., “Summarize in 2–3 sentences, specific to the product”).  
- **Retriever (k value)**: Currently using k=3. Will test k=5 to see if answers become more complete.  
- **Generator model**: Using flan‑t5‑base. Will compare with flan‑t5‑small for speed and instruction following.  
- **Sample size**: Reduced to 1000 complaint chunks to keep memory usage low. Larger samples caused freezing.  
