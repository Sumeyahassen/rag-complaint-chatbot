import pandas as pd

def clean_complaints(input_path, output_path, chunksize=100000):
    cleaned_chunks = []

    # Read in chunks
    for chunk in pd.read_csv(
        input_path,
        dtype={"Consumer disputed?": str},
        low_memory=False,
        chunksize=chunksize
    ):
        # Drop empty narratives
        chunk = chunk[chunk['Consumer complaint narrative'].notna()]
        chunk = chunk[chunk['Consumer complaint narrative'].str.strip() != ""]

        # Add narrative length column
        chunk['narrative_length'] = chunk['Consumer complaint narrative'].str.split().str.len()

        cleaned_chunks.append(chunk)

    # Combine all cleaned chunks
    df = pd.concat(cleaned_chunks)

    # Save cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"Processed dataset saved to {output_path}")
    return df

if __name__ == "__main__":
    clean_complaints(
        "./data/raw/complaints.csv",
        "./data/processed/filtered_complaints.csv"
    )
