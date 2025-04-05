import os
import yaml
from rich.console import Console
from rich.text import Text


def validate_config(config: dict):
    if not isinstance(config["model"], str):
        raise ValueError("Model name must be a string.")
    if not isinstance(config["custom_model"], str):
        raise ValueError("Custom model name must be a string.")
    if not os.path.isdir(config["root_dir_path"]):
        raise FileNotFoundError("Root directory path not found.")
    if not os.path.isdir(config["csv_corpus_path"]):
        raise FileNotFoundError("Csv corpus path not found.")
    if not isinstance(config["model_annotations_save_path"], str):
        raise ValueError("Model annotation save path must be a string.")
    if not isinstance(config["results_save_path"], str):
        raise ValueError("Results save path must be a string.")
    if not isinstance(config["skip_already_annotated"], bool):
        raise ValueError("skip_already_annotated must be a boolean.")
    if not isinstance(config["save_result_metrics"], bool):
        raise ValueError("save_result_metrics must be a boolean.")
    if not isinstance(config["save_result_matrices"], bool):
        raise ValueError("save_result_matrices must be a boolean.")


def display_config(config: dict):
    console = Console()
    console.print("\n[bold cyan]Configuration Settings:[/bold cyan]\n")

    for key, value in config.items():
        key_text = Text(key.replace("_", " ").capitalize())
        console.print(f"[bold blue]{key_text}[/bold blue]: {value}")
    console.print("\n")


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as inf:
        config = yaml.safe_load(inf)
    validate_config(config)
    display_config(config)
    return config
