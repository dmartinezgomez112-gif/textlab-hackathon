import pandas as pd
from datasets import load_dataset

def main():
    print("Descargando Multilingual Financial Sentiment desde Hugging Face...")
    ds = load_dataset("Kenpache/multilingual-financial-sentiment", split="train")
    df = ds.to_pandas()

    lang_cols = [c for c in df.columns if "lang" in c.lower()]
    if lang_cols:
        lang_col = lang_cols[0]
        df_es = df[df[lang_col].astype(str).str.lower().isin(["es", "spanish"])]
    else:
        print("Columnas encontradas:", df.columns.tolist())
        df_es = df

    output_file = "financial_sentiment_es.csv"
    df_es.to_csv(output_file, index=False)
    print(f"Dataset guardado exitosamente en '{output_file}' con {len(df_es)} registros.")

if __name__ == "__main__":
    main()
