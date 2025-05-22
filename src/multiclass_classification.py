import os
import ollama
import pandas as pd
from tqdm import tqdm

from evaluation import evaluate_annotation
from load_config import Config, ClassificationTypes
from categories import Categories
from evaluation import Metrics, pretty_print


CONFIG = Config(print_config=False)


def annotate_with_ollama_multiclass(sentences: list[str]) -> list:
    results = []
    for sentence in tqdm(sentences, desc="Annotation en cours"):
        response = ollama.chat(
            model=CONFIG.custom_model, messages=[{"role": "user", "content": sentence}]
        )
        annotation = response["message"]["content"].strip().lower()

        if annotation in [cat.value for cat in Categories]:
            results.append(annotation)
        else:
            results.append(-1)

    return results


def get_annotated_df_multiclass(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file, sep="|")
    sentences = df["dynamique_text"].tolist()
    annotations = annotate_with_ollama_multiclass(sentences)
    df["annotation"] = annotations

    print("INFO : Suppression des annotations invalides")
    nb_of_invalid = len(df[df["annotation"] == -1])
    print(f"{nb_of_invalid} lignes sont invalides et donc supprimees.")
    df = df[df["annotation"] != -1]

    if CONFIG.save_annotations:
        annotation_path = os.path.join(
            CONFIG.root_dir_path, CONFIG.model_annotations_save_path, CONFIG.model
        )
        os.makedirs(annotation_path, exist_ok=True)
        df.to_csv(
            f"{annotation_path}/{os.path.basename(csv_file)}", sep="|", index=False
        )

    return df


def save_results_multiclass(metrics: Metrics, conf_matrix, filename):
    if CONFIG.save_result_metrics:
        result_path = os.path.join(CONFIG.result_save_path, CONFIG.model)
        os.makedirs(result_path, exist_ok=True)
        metrics_df = pd.DataFrame(
            {
                "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
                "Score": [
                    metrics.accuracy,
                    metrics.precision,
                    metrics.recall,
                    metrics.f1,
                ],
            }
        )
        metrics_df.to_csv(f"{result_path}/{filename}", index=False)

    if CONFIG.save_result_matrices:
        matrices_path = os.path.join(
            CONFIG.root_dir_path, CONFIG.result_save_path, CONFIG.model, "matrices"
        )
        os.makedirs(matrices_path, exist_ok=True)
        matrix_df = pd.DataFrame(conf_matrix, columns=[cat.value for cat in Categories])
        matrix_df.index = [cat.value for cat in Categories]
        pd.DataFrame(matrix_df).to_csv(
            f"{matrices_path}/{filename.split('.')[0]}_confusion_matrix.csv"
        )


def perform_multiclass_classification(csv_file: str, filename: str):
    annotated_df = get_annotated_df_multiclass(csv_file)
    evaluation, conf_matrix = evaluate_annotation(
        annotated_df, ClassificationTypes.Multiclass
    )
    pretty_print(evaluation, conf_matrix, filename, type=ClassificationTypes.Multiclass)
    if CONFIG.save_result_metrics or CONFIG.save_result_matrices:
        save_results_multiclass(evaluation, conf_matrix, filename)
