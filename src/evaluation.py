from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from collections import namedtuple
import pandas as pd

from load_config import ClassificationTypes
from categories import Categories

Metrics = namedtuple("Metrics", ["accuracy", "precision", "recall", "f1"])


def evaluate_annotation(annotated_df, type: ClassificationTypes):
    if type == ClassificationTypes.Binary:
        y_true = annotated_df["dynamic"].tolist()
    else:
        y_true = annotated_df["dynamique_type"].tolist()
    y_pred = annotated_df["annotation"].tolist()
    accuracy = accuracy_score(y_true, y_pred)
    if type == ClassificationTypes.Binary:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary"
        )
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro"
        )
    metrics = Metrics(accuracy, precision, recall, f1)
    conf_matrix = confusion_matrix(
        y_true, y_pred, labels=[cat.value for cat in Categories]
    )
    return metrics, conf_matrix


def pretty_print(
    metrics: Metrics, conf_matrix, filename, type: ClassificationTypes
) -> None:
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
    if type == ClassificationTypes.Binary:
        conf_table = Table(header_style="bold red")
        conf_table.add_column("")
        conf_table.add_column("Predicted False", justify="center", style="bold green")
        conf_table.add_column("Predicted Dynamic", justify="center", style="bold green")
        conf_table.add_row(
            "Actual False", str(conf_matrix[0][0]), str(conf_matrix[0][1])
        )
        conf_table.add_row(
            "Actual Dynamic", str(conf_matrix[1][0]), str(conf_matrix[1][1])
        )

        console.print(conf_table)

    else:
        class_labels = [cat.value for cat in Categories]

        conf_table = Table(header_style="bold red")
        conf_table.add_column("Actual \ Predicted", style="bold blue")

        # Add predicted class columns
        for label in class_labels:
            conf_table.add_column(label, justify="center", style="bold green")

        # Add rows
        for i, row in enumerate(conf_matrix):
            row_label = class_labels[i]
            row_values = [str(cell) for cell in row]
            conf_table.add_row(row_label, *row_values)

        console.print(conf_table)


if __name__ == "__main__":
    annotated_df = pd.read_csv(
        "annotations/annotations_llm/deepseek-r1:1.5b/1958-padog_CE_Oct23_GP.csv",
        sep="|",
    )
    metrics, conf_matrix = evaluate_annotation(annotated_df, ClassificationTypes.Binary)
    pretty_print(
        metrics,
        conf_matrix,
        "annotations/annotations_llm/deepseek-r1:1.5b/1958-padog_CE_Oct23_GP.csv",
        type=ClassificationTypes.Binary,
    )
