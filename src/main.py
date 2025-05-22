import os

import glob
from binary_classification import perform_binary_classification
from multiclass_classification import perform_multiclass_classification
from load_config import ClassificationTypes, Config

CONFIG = Config()


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
        if CONFIG.classification == ClassificationTypes.Binary:
            perform_binary_classification(csv_file, filename)
        else:
            perform_multiclass_classification(csv_file, filename)


if __name__ == "__main__":
    main()
