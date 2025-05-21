import os
import yaml
from rich.console import Console
from rich.text import Text


class Config:
    def __init__(self, config_path="config.yaml", print_config=True):
        self.as_dict: dict = self.load_config(config_path)
        self.model: str = self.as_dict["model"]
        self.task: str = self.as_dict["task"]
        self.custom_model: str = self.as_dict["custom_model"]
        self.root_dir_path: str = self.as_dict["root_dir_path"]
        self.csv_corpus_path: str = self.as_dict["csv_corpus_path"]
        self.model_annotations_save_path: str = self.as_dict[
            "model_annotations_save_path"
        ]
        self.result_save_path: str = self.as_dict["results_save_path"]
        self.skip_already_annotated: bool = self.as_dict["skip_already_annotated"]
        self.save_annotations: bool = self.as_dict["save_annotations"]
        self.save_result_metrics: bool = self.as_dict["save_result_metrics"]
        self.save_result_matrices: bool = self.as_dict["save_result_matrices"]
        if print_config:
            self.display_config()

    def load_config(self, config_path):
        with open(config_path, "r") as inf:
            config = yaml.safe_load(inf)
        self.validate_config(config)
        return config

    @staticmethod
    def validate_config(config: dict):
        if not isinstance(config["model"], str):
            raise ValueError("Model name must be a string.")
        if not isinstance(config["custom_model"], str):
            raise ValueError("Custom model name must be a string.")
        if not isinstance(config["task"], str):
            raise ValueError("Task name must be a string.")
        if not os.path.isdir(config["root_dir_path"]):
            raise FileNotFoundError("Root directory path not found.")
        if not os.path.isdir(config["csv_corpus_path"]):
            raise FileNotFoundError("Csv corpus path not found.")
        if not isinstance(config["model_annotations_save_path"], str):
            raise ValueError("Model annotation save path must be a string.")
        if not isinstance(config["results_save_path"], str):
            raise ValueError("Results save path must be a string.")
        if not isinstance(config["save_annotations"], bool):
            raise ValueError("save_annotations must be a boolean.")
        if not isinstance(config["skip_already_annotated"], bool):
            raise ValueError("skip_already_annotated must be a boolean.")
        if not isinstance(config["save_result_metrics"], bool):
            raise ValueError("save_result_metrics must be a boolean.")
        if not isinstance(config["save_result_matrices"], bool):
            raise ValueError("save_result_matrices must be a boolean.")

    def display_config(self):
        console = Console()
        console.print("\n[bold cyan]Configuration Settings:[/bold cyan]\n")

        for key, value in self.as_dict.items():
            key_text = Text(key.replace("_", " ").capitalize())
            console.print(f"[bold blue]{key_text}[/bold blue]: {value}")
        console.print("\n")
