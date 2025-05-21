import os
import pandas as pd
import glob
from evaluation import Metrics, evaluate_annotation, pretty_print

from load_config import Config
from dyanmic_annotation import run_annotation


CONFIG = Config()


def save_results(metrics: Metrics, conf_matrix, filename):
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
        conf_matrix_df = pd.DataFrame(
            conf_matrix, columns=["Predicted False", "Predicted Dynamic"]
        )
        conf_matrix_df.index = ["Actual False", "Actual Dynamic"]
        matrices_path = os.path.join(
            CONFIG.root_dir_path, CONFIG.result_save_path, CONFIG.model, "matrices"
        )
        os.makedirs(matrices_path, exist_ok=True)
        conf_matrix_df.to_csv(
            f"{matrices_path}/{filename.split('.')[0]}_confusion_matrix.csv"
        )


def main():
    csv_path = os.path.join(CONFIG.root_dir_path, CONFIG.csv_corpus_path)
    all_csv_files = glob.glob(f"{csv_path}/*.csv")
    for csv_file in all_csv_files:
        filename = csv_file.split("/")[-1]
        print(f"Currently loading {filename}...")
        if CONFIG.skip_already_annotated and os.path.isfile(
            f"{CONFIG.result_save_path}/{CONFIG.model}/{filename}"
        ):
            continue
        if CONFIG.task == "annotation":
            annotated_df = run_annotation(csv_file)
            evaluation, conf_matrix = evaluate_annotation(annotated_df)
        # else:
        # annotated_df = run_extraction(csv_file)
        # evaluation, conf_matrix = evaluate_extraction(annotated_df)
        pretty_print(evaluation, conf_matrix, filename)
        if CONFIG.save_result_metrics or CONFIG.save_result_matrices:
            save_results(evaluation, conf_matrix, filename)


if __name__ == "__main__":
    main()
