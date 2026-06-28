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

# Task‑4 Evaluation

# Task‑4 Evaluation

| Question | Answer | Retriever k=3 | Retriever k=5 | Quality | Notes |
|----------|--------|---------------|---------------|---------|-------|
| What are the most common credit card complaints? | late fees and damaging credit reports | late fees and damaging credit reports | late fees and damaging credit reports | 4 | Consistent, relevant |
| What issues do consumers report about loans? | late fees and damaging credit reports | late fees and damaging credit reports | late fees and damaging credit reports | 2 | Too generic |
| Are there complaints about savings accounts? | Yes | Yes | account information | 3 | Mixed, vague |
| What problems occur with money transfers? | Detailed narrative | Detailed narrative | shorter narrative | 3 | Informative but too long |
| Question | Answer | Quality | Notes |
|----------|--------|---------|-------|
| What are the most common credit card complaints? | late fees and damaging credit reports | 4 | Clear and relevant |
| What issues do consumers report about loans? | late fees and damaging credit reports | 2 | Too generic, not loan‑specific |
| Are there complaints about savings accounts? | Yes | 3 | Vague, needs detail |
| What problems occur with money transfers? | Long narrative about blocked/stolen transfers | 3 | Informative but too long |
| Do consumers report problems with customer service? | No | 2 | Likely incomplete |
| Are there complaints about interest rates? | Yes | 3 | Too short, needs context |
 to run  this command
 streamlit run app.py


