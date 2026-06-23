def clean_complaints(df):
    df = df[df['Consumer complaint narrative'].notna()]
    df = df[df['Consumer complaint narrative'].str.strip() != ""]
    df['narrative_length'] = df['Consumer complaint narrative'].str.split().str.len()
    return df
