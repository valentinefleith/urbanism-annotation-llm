import os
import ollama
import pandas as pd
from collections import namedtuple
import glob
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from rich.console import Console
from rich.table import Table

MODEL = "mistral"
CSV_PATH = "corpus/corpus_phrases"
ANNOTATIONS_PATH = f"annotations/annotations_llm/{MODEL}"
RESULTS_PATH = f"results/classification/{MODEL}"

Metrics = namedtuple("Metrics", ["accuracy", "precision", "recall", "f1"])


def annotate_with_ollama(sentences: pd.DataFrame) -> list:
    results = []
    for sentence in sentences:
        response = ollama.chat(
            model="urbaniste", messages=[{"role": "user", "content": sentence}]
        )
        annotation = response["message"]["content"]
        results.append(int(annotation))
    return results


def get_annotated_df(csv_file: str, save=True) -> pd.DataFrame:
    df = pd.read_csv(csv_file, sep="|")
    sentences = df["sentence"].tolist()
    annotations = annotate_with_ollama(sentences)
    df["annotation"] = annotations
    if save:
        os.makedirs(ANNOTATIONS_PATH, exist_ok=True)
        df.to_csv(f"{ANNOTATIONS_PATH}/{csv_file.split('/')[-1]}", sep="|", index=False)
    return df


def evaluate_annotation(annotated_df: pd.DataFrame):
    y_true = annotated_df["dynamic"].tolist()
    y_pred = annotated_df["annotation"].tolist()
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    metrics = Metrics(accuracy, precision, recall, f1)
    conf_matrix = confusion_matrix(y_true, y_pred)
    return metrics, conf_matrix


def pretty_print(metrics: Metrics, conf_matrix, filename) -> None:
    console = Console()
    # METRICS
    console.print(f"\n[bold cyan]Results for {filename}[/bold cyan]\n")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="bold yellow")

    table.add_row("Accuracy", f"{metrics.accuracy:.4f}")
    table.add_row("Precision", f"{metrics.precision:.4f}")
    table.add_row("Recall", f"{metrics.recall:.4f}")
    table.add_row("F1-score", f"{metrics.f1:.4f}")

    console.print(table)

    # CONFUSION MATRIX
    console.print("\n[bold cyan]Confusion Matrix[/bold cyan]")
    conf_table = Table(header_style="bold red")
    conf_table.add_column("")
    conf_table.add_column("Predicted False", justify="center", style="bold green")
    conf_table.add_column("Predicted Dynamic", justify="center", style="bold green")

    conf_table.add_row("Actual False", str(conf_matrix[0][0]), str(conf_matrix[0][1]))
    conf_table.add_row("Actual Dynamic", str(conf_matrix[1][0]), str(conf_matrix[1][1]))

    console.print(conf_table)


def save_results(metrics: Metrics, conf_matrix, filename):
    os.makedirs(RESULTS_PATH, exist_ok=True)
    metrics_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
            "Score": [metrics.accuracy, metrics.precision, metrics.recall, metrics.f1],
        }
    )
    metrics_df.to_csv(f"{RESULTS_PATH}/{filename}", index=False)
    conf_matrix_df = pd.DataFrame(
        conf_matrix, columns=["Predicted False", "Predicted Dynamic"]
    )
    conf_matrix_df.index = ["Actual False", "Actual Dynamic"]
    conf_matrix_df.to_csv(f"{RESULTS_PATH}/{filename}_confusion_matrix")


def main():
    all_csv_files = glob.glob(CSV_PATH + "/*.csv")
    for csv_file in all_csv_files:
        filename = csv_file.split("/")[-1]
        annotated_df = get_annotated_df(csv_file)
        evaluation, conf_matrix = evaluate_annotation(annotated_df)
        pretty_print(evaluation, conf_matrix, filename)
        save_results(evaluation, conf_matrix, filename)


if __name__ == "__main__":
    main()
