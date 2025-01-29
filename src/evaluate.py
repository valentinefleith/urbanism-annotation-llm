import pandas as pd
import glob


# DIMENSIONS = ["dyn", "act", "perc", "tps", "doc"]
DIMENSIONS = ["dyn"]
ANNOTATIONS_AUTO = "annotations/annotations_auto"
ANNOTATIONS_MANUELLES = "annotations/annotations_manuelles"
SEUIL_SIMILARITE = 0.5


def jaccard_similarity(text1, text2):
    """Calcule la similarité de Jaccard entre deux phrases (intersection / union des mots)."""
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0


def dynamic_jaccard_threshold(text1, text2):
    """Ajuste dynamiquement le seuil de Jaccard en fonction de la longueur"""
    len_text1 = len(text1.split())
    len_text2 = len(text2.split())
    avg_length = (len_text1 + len_text2) / 2

    if avg_length < 5:
        return 0.8  # Si phrases courtes, exige plus de similarité
    elif avg_length < 10:
        return 0.6
    else:
        return 0.5  # Si phrases longues, accepte un seuil plus bas


def normalize_text(text):
    if isinstance(text, str):
        return text.strip().lower()
    return ""


def evaluate_matching(segments_bert, segments_reels, dim):
    # Normalisation et gestion des NaN
    segments_reels = segments_reels.copy()
    segments_bert = segments_bert.copy()

    segments_reels.loc[:, "Segment_Annoté"] = (
        segments_reels["Segment_Annoté"].astype(str).apply(normalize_text)
    )
    segments_bert.loc[:, "balise_dyn"] = (
        segments_bert[f"balise_{dim}"].astype(str).apply(normalize_text)
    )

    segments_reels_list = segments_reels["Segment_Annoté"].tolist()

    # Déterminer les segments correctement détectés (TP)
    matching_rows = segments_bert[
        segments_bert[f"balise_{dim}"].apply(
            lambda x: any(
                jaccard_similarity(x, s) >= dynamic_jaccard_threshold(x, s)
                for s in segments_reels_list
            )
        )
    ]

    TP = len(matching_rows)  # Vrais positifs
    FP = len(segments_bert) - TP  # Faux positifs
    FN = len(segments_reels) - TP  # Faux négatifs

    # Précision, Rappel, F-mesure
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print(f"Précision : {precision:.4f}")
    print(f"Rappel    : {recall:.4f}")
    print(f"F-mesure  : {f1_score:.4f}")

    return precision, recall, f1_score


def compute_evaluation(filename):
    name_in_table = f"{filename.split('/')[-1].split('.')[0]}.xml"
    for dim in DIMENSIONS:  # TODO: handle dyn and other dimensions separately
        print(f"CURRENTLY DEALING WITH: {dim}")
        ground_truth_df = pd.read_csv(
            f"{ANNOTATIONS_MANUELLES}/all_{dim}.tsv", sep="\t"
        ).query("`Nom_fichier` == @name_in_table")
        annotation_auto = pd.read_csv(filename)
        segments_reels = ground_truth_df[["Segment_Annoté"]]
        segments_bert = annotation_auto[[f"balise_{dim}", "phrase", "phrase_id"]]
        # print(f"SEGMENTS REEELS:\n{segments_reels}\n")
        # print(f"SEGMENTS EXTRAITS PAR BERT:\n{segments_bert}")
        # accuracy_score = get_accuracy_score(segments_bert, segments_reels)
        # print(accuracy_score)
        metrics = evaluate_matching(segments_bert, segments_reels, dim)
        print(metrics)

    return 1


def main():
    all_files = glob.glob(f"{ANNOTATIONS_AUTO}/*.tsv")
    accuracy_scores = []
    for file in all_files:
        accuracy_scores.append(compute_evaluation(file))
    print(accuracy_scores)


if __name__ == "__main__":
    main()
