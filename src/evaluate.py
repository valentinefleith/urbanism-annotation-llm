import pandas as pd
import glob


# DIMENSIONS = ["dyn", "acts", "lieux", "perc", "tps", "docs"]
DIMENSIONS = ["dyn"]
ANNOTATIONS_AUTO = "annotations/annotations_auto"
ANNOTATIONS_MANUELLES = "annotations/annotations_manuelles"


def compute_accuracy_score(filename):
    name_in_table = f"{filename.split('/')[-1].split('.')[0]}.xml"
    for dim in DIMENSIONS:  # TODO: handle dyn and other dimensions separately
        ground_truth_df = pd.read_csv(
            f"{ANNOTATIONS_MANUELLES}/all_{dim}.tsv", sep="\t"
        ).query("`Nom_fichier` == @name_in_table")
        annotation_auto = pd.read_csv(filename)
        print(annotation_auto)
        # calculer rappel et precision separement
        # verifier alignement
        # calculer fmesure

    return 1


def main():
    all_files = glob.glob(f"{ANNOTATIONS_AUTO}/*.tsv")
    accuracy_scores = []
    for file in all_files:
        accuracy_scores.append(compute_accuracy_score(file))
    print(accuracy_scores)


if __name__ == "__main__":
    main()
