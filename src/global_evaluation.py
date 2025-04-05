import pandas as pd
import glob
import statistics as s
from rich.console import Console
from rich.table import Table


RESULTS_PATH = "results/classification/first_tests"  # for testing
# RESULTS_PATH = "results/classification/promptv1 # uncomment when we have new results


class Results:
    def __init__(self, model: str, files: list[str]):
        self.model: str = model
        self.files: list[str] = self.get_related_files(files)

    def get_related_files(self, files: list[str]):
        return [file for file in files if file.split("/")[-2] == self.model]

    def compute_avg_score(self, score: str):
        return s.mean(
            [
                pd.read_csv(file).query(f"`Metric` == '{score}'")["Score"].to_list()[0]
                for file in self.files
            ]
        )

    @property
    def avg_accuracy(self):
        return self.compute_avg_score("Accuracy")

    @property
    def avg_precision(self):
        return self.compute_avg_score("Precision")

    @property
    def avg_recall(self):
        return self.compute_avg_score("Recall")

    @property
    def avg_f1(self):
        return self.compute_avg_score("F1-Score")

    def __repr__(self):
        return (
            f"model={self.model}, "
            f"avg_accuracy={self.avg_accuracy}, "
            f"avg_precision={self.avg_precision}, "
            f"avg_recall={self.avg_recall}, "
            f"avg_f1={self.avg_f1}"
        )


def get_model_names(results_files: list[str]) -> list[str]:
    return list(set([file.split("/")[-2] for file in results_files]))


def compute_global_results(result_files, model_names) -> pd.DataFrame:
    results = [Results(model, result_files) for model in model_names]
    compared_results = pd.DataFrame(
        {
            result.model: [
                result.avg_accuracy,
                result.avg_precision,
                result.avg_recall,
                result.avg_f1,
            ]
            for result in results
        },
        index=pd.Index(["Accuracy", "Precision", "Recall", "F1-score"]),
    )
    return compared_results


def pretty_print(compared_results: pd.DataFrame):
    console = Console()

    if compared_results.empty:
        console.print("[bold red]No results to display.[/bold red]")
        return

    table = Table(
        title="Model Comparison Results", show_header=True, header_style="bold cyan"
    )

    table.add_column("Metric", style="bold magenta")
    for model in compared_results.columns:
        table.add_column(model, justify="center", style="green")

    for metric, values in compared_results.iterrows():
        table.add_row(metric, *[f"{v:.4f}" for v in values])

    console.print(table)


def main():
    results_files = glob.glob(f"{RESULTS_PATH}/*/*.csv")
    model_names = get_model_names(results_files)
    if not model_names:
        print("No results to display.")
    global_results = compute_global_results(results_files, model_names)
    pretty_print(global_results)
    global_results.to_csv(f"{RESULTS_PATH}/global.csv")


if __name__ == "__main__":
    main()
