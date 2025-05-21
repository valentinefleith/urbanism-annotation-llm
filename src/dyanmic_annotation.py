import os
import ollama
import pandas as pd
from tqdm import tqdm
from sklearn.utils import resample

from load_config import Config

CONFIG = Config(print_config=False)


def annotate_with_ollama(sentences: pd.DataFrame) -> list:
    results = []
    for sentence in tqdm(sentences, desc="Annotation en cours"):
        response = ollama.chat(
            model=CONFIG.custom_model, messages=[{"role": "user", "content": sentence}]
        )
        annotation = response["message"]["content"].strip()

        if CONFIG.model.startswith("deepseek"):
            print(f"🟡 DEBUG - Réponse brute du modèle : {annotation}")
        try:
            annotation = (
                annotation[-1] if CONFIG.model.startswith("deepseek") else annotation[0]
            )
            if int(annotation) in [0, 1]:
                results.append(int(annotation))
            else:
                results.append(-1)
        except (ValueError, IndexError):
            print(f"⚠️ Erreur : Réponse inattendue -> {annotation}")
            results.append(0)

    return results


def get_downsampled(df) -> pd.DataFrame:
    class_counts = df["dynamic"].value_counts()
    biggest_class = class_counts.idxmax()
    # we separate the majority class from the rest of the samples
    biggest_class_df = df.query(f"`dynamic` == {biggest_class}")
    df_without_biggest = df.query(f"`dynamic` != {biggest_class}")
    resampled_class = resample(
        biggest_class_df,
        replace=False,
        n_samples=int(class_counts.median()),  # reduce n_sample to median
        random_state=42,
    )
    # and then concatenate them after reducing the size
    return (
        pd.concat([df_without_biggest, resampled_class])
        .sample(frac=1)
        .reset_index(drop=True)
    )


def run_annotation(csv_file: str) -> pd.DataFrame:
    """
    Charge un fichier CSV, applique un downsampling et annote les phrases.
    """
    df = pd.read_csv(csv_file, sep="|")
    downsampled_df = get_downsampled(df)
    sentences = downsampled_df["sentence"].tolist()
    annotations = annotate_with_ollama(sentences)
    downsampled_df["annotation"] = annotations

    print("🟡 INFO : Suppression des annotations invalides")
    nb_of_invalid = len(downsampled_df[downsampled_df["annotation"] == -1])
    print(f"{nb_of_invalid} lignes sont invalides et donc supprimees.")
    downsampled_df = downsampled_df[downsampled_df["annotation"] != -1]

    if CONFIG.save_annotations:
        annotation_path = os.path.join(
            CONFIG.root_dir_path, CONFIG.model_annotations_save_path, CONFIG.model
        )
        os.makedirs(annotation_path, exist_ok=True)
        downsampled_df.to_csv(
            f"{annotation_path}/{os.path.basename(csv_file)}", sep="|", index=False
        )

    return downsampled_df
