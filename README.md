[![image](https://github.com/user-attachments/assets/d7aeeb94-365a-48e0-98db-ca412f5a36a4)](https://ollama.com/)


# Automatic annotation of urban dynamics with LLMs: a comparative approach

## Overview
This repo is part of a **university-supervised research project** on automatic text annotation in the field of urbanism.  
It leverages **Ollama** and **Mistral/Llama3/Deepseek** to process input text and detect urban dynamics based on predefined annotation rules.

The model is configured using a **Modelfile** located in `prompt-playground/config/`, which defines the system prompt and behavior for urbanism-focused annotation.

---

## Prerequisites

- **Python** >= 3.10
- A **UNIX** os or Windows with **WSL**
- **Ollama** following the [documentation](https://ollama.com/download)


## Installation

### 1. Clone the Repository
```bash
git clone git@github.com:valentinefleith/urbanism-annotation-llm.git
cd urbanism-annotation-llm
```
### 2. Install Dependencies

```bash
make setup
```

or just

```bash
make
```

### 3. Install the required model on Ollama
Before running the project, you have to install the LLM model that you want. For example:
```bash
ollama install mistral
```
The model behavior is defined in a `Modelfile` config file. There are a few examples in the `prompt-playground/prompts` directory. This Modelfile controls the **system prompt** and how the model processes urbanism-related texts.

At the top of the `Modelfile`, make sure to configure properly the name of the model that you just installed.

For example, if you want to use another model than Mistral, you can update it in the `Modelfile` by changing the `FROM` line:
```
FROM <your-new-model>
```

Finally, to build the model, type:
```bash
ollama create urbaniste -f ./prompt-playground/config/Modelfile
```
If you change the model name (something else that `urbaniste`, make sure to update it properly in the `config.yaml` file (cf. section 4).

### 4. Configuration
You have to modify `config.yaml` file according to your settings. Here are the required fields:
```yaml
# MODEL INFORMATION
model: "deepseek-r1:1.5b"
custom_model: "urbaniste"

# TASK
classification: "binary" # "binary" or "multiclass"

# PATH-RELATED INFORMATION
root_dir_path: "/Users/valentinefleith/Perso/code/projet-tuteure"
csv_corpus_path: "corpus/corpus_phrases"
model_annotations_save_path: "annotations/annotations_llm/promptv1"
results_save_path: "results/classification/promptv1"

# BOOLEAN INFORMATION
save_annotations: true
skip_already_annotated: true
save_result_metrics: true
save_result_matrices: true
```

This allows you to run this pipeline this any corpus that you want. The corpus paths must refer to a directory containing csv files, in this format:

- _For binary classification:_

| sentence | dynamic |
|----------|---------|
| C’est l’objectif que le Gouvernement assignait le 31 décembre 1958 au plan d’Aménagement et d’Organisation Générale de la Région Parisienne. | 0 |
| Des milliers de logements sans confort se sont édifiés dans un incroyable désordre. | 1 |

The separator is a pipe (`|`) and each class is representend by either `0` or `1`.

- _For multiclass classification:_

| dynamique_text                                                                 | dynamique_type |
|--------------------------------------------------------------------------------|----------------|
| se sont édifiés                                                                | crea           |
| l’embellissement de Paris                                                     | modif          |
| n’ont ajouté à « la plus belle ville du monde » qu’une banlieue indifférenciée, sans style et sans âme | crea           |
| se prépara puis s’accomplit le renouveau de la France                         | repr           |

The separator is a pipe (`|`) and each class is representend by a label , which must be updated in [`src/categories.py`](https://github.com/valentinefleith/urbanism-annotation-llm/blob/main/src/categories.py).

```py
from enum import Enum


class Categories(Enum):
    Creation = "crea"
    Destruction = "destr"
    Modification = "modif"
    Reprise = "repr"
    Maintien = "maint"
```

NB: to get `root_dir_path`, just type `pwd` in a terminal from the root of this repository and paste the result.

_NB2: Configuration is in progress, you should soon configure class names in config file._


### 5. Run the project
Once the installation is complete, run:

```bash
make run
```
This executes `main.py` which:
- Loads the urbanism-focused LLM (`urbaniste`)
- Processes text and **classifies urban dynamics**
- Saves LLM annotation in `annotations/annotations_llm/{model-name}`
- Evaluates the model and saves results in `results/classification/{model-name}`

During the running, it prints the result tables which are also saved. For example for binary classification:

![image](https://github.com/user-attachments/assets/159eb060-5526-46ba-8373-26a5400ce66f)

And for multiclass:

![image](https://github.com/user-attachments/assets/93b5adf9-b727-46cf-802b-31317fbb24eb)



Once you tested on any model you want, you can compare the results:
```bash
make eval
```
Example output:

<img width="472" alt="image" src="https://github.com/user-attachments/assets/bb71d538-7867-4f3f-bf7a-97a434966e96" />

---
### TODO:
- Improve annotation accuracy by refining prompts
- Support multiple annotation types beyond urban dynamics
