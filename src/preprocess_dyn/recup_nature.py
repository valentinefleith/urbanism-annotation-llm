# -*- coding: utf-8 -*-
import pandas as pd
import os

CSV_PATH = "nature.csv" 
OUTPUT_DIR = "nature_dyn"


def recup_nature():
    """
    Script qui récupère le texte de la balise dynamique ainsi que l'information sur la nature de la dynamique.
    input : fichier.csv de format : fichier|dynamique_brut|dynamique_text|dynamique_type
    output : dossier contenant des fichiers csv nommés par la colonne 'fichier' du csv d'entrée,
             avec colonnes : dynamique_text, dynamique_type
             - sans doublons
             - sans les types "abs"
             - avec un maximum de 430 exemples par type de dynamique dans chaque fichier
    """
    df = pd.read_csv(CSV_PATH, sep="|")
    df = df[["fichier", "dynamique_text", "dynamique_type"]]

    #on garde tous les types sauf "ab"
    df = df[df["dynamique_type"] != "ab"]

    #remplacement les abréviations par les noms complets 
    df.loc[df["dynamique_type"] == "cre",  "dynamique_type"] = "crea"
    df.loc[df["dynamique_type"] == "dest", "dynamique_type"] = "destr"
    df.loc[df["dynamique_type"] == "modi", "dynamique_type"] = "modif"
    df.loc[df["dynamique_type"] == "main", "dynamique_type"] = "maint"
    df.loc[df["dynamique_type"] == "rep",  "dynamique_type"] = "repr"

    # créer dossier output s'il n'existe pas
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # on parcourt chaque fichier
    for nom_fichier, group in df.groupby("fichier"):
        # supprimer les doublons dans le groupe courant
        group = group.drop_duplicates(subset=["dynamique_text", "dynamique_type"])

        # limiter à 430 par type
        group_equilibre = group.groupby("dynamique_type").head(430)

        # ne garder que texte et type de la dynamique
        sortie = group_equilibre[["dynamique_text", "dynamique_type"]]

        # sauvegarder dans output/nom_fichier.csv
        nom_fichier = nom_fichier.split(".")[0]
        chemin_sortie = os.path.join(OUTPUT_DIR, f"{nom_fichier}.csv")
        sortie.to_csv(chemin_sortie, index=False, sep="|")

        print(f"Fichier {chemin_sortie} créé.")


recup_nature()