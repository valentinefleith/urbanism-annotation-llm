# Première étape: évaluation de l'outil

- L'outil utilisé BERT qui a été fine-tuné pour annoter automatiquement les dynamiques urbaines. On a récupéré de travail du groupe de Jinyu de l'année dernière pour obtenir un corpus bien annoté, on a lancé les textes bruts dans l'outil et on a évalué les résultats.


On a utilisé la similarité de Jaccard.

Pour l'instant premiers résultats assez mauvais:

| Dimension | Précision  | Rappel     | F-mesure   |
|-----------|------------|------------|------------|
| act       | 0.0        | 0.0        | 0.0        |
| doc       | 0.0125     | 0.4651     | 0.0243     |
| dyn       | 0.1891     | 0.2834     | 0.2269     |
| lieuloc   | 0.0        | 0.0        | 0.0        |
| lieuobj   | 0.0        | 0.0        | 0.0        |
| perc      | 0.0        | 0.0        | 0.0        |
| tps       | 0.0        | 0.0        | 0.0        |
