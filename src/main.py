import os
import ollama
import pandas as pd
import glob

CSV_PATH = "corpus/corpus_phrases"
ANNOTATIONS_PATH = "annotations/annotations_llm/mistral"


def annotate_with_ollama(sentences: pd.DataFrame) -> list:
    results = []
    for sentence in sentences:
        response = ollama.chat(
            model="urbaniste", messages=[{"role": "user", "content": sentence}]
        )
        annotation = response["message"]["content"]
        results.append(annotation)
    return results


def main():
    all_csv_files = glob.glob(CSV_PATH + "/*.csv")
    for csv_file in all_csv_files:
        df = pd.read_csv(csv_file, sep="|")
        sentences = df["sentence"].tolist()
        annotations = annotate_with_ollama(sentences)
        df["annotation"] = annotations
        os.makedirs(ANNOTATIONS_PATH, exist_ok=True)
        df.to_csv(f"{ANNOTATIONS_PATH}/{csv_file.split('/')[-1]}", sep="|", index=False)


if __name__ == "__main__":
    main()
