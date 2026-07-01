# A - Architecture de ce BonusTrack °10 
## Arborescence Structurante 
> _"mini-sous Livrable"_
```python
project/
│
├── data/
│   └── support_it.db
│
├── get_from_database.py
├── transform_to_gold.py
├── train.py
│
├── requirements.txt
├── Dockerfile.extract
├── Dockerfile.train
├── docker-compose.yml
│
├── extracted.jsonl
├── gold_data.jsonl
└── it_assistant_finetuned/
```

# B - Conteneur n°1 : extraction des données
## Dockerfile.extract

```python
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY get_from_database.py .

ENTRYPOINT ["python", "get_from_database.py"]
```

# C - Conteneur n°2 : fine-tuning LoRA
## Dockerfile.train

```python

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

ENTRYPOINT ["python", "train.py"]
```
> ℹ️ Les principes Docker (Dockerfile, image Python, installation des dépendances, répertoire de travail et commande de démarrage) sont conformes aux pratiques documentées pour la conteneurisation d'applications Python.

# D - requirements.txt 
```python
torch
transformers
datasets
peft
trl
accelerate
sentencepiece
protobuf
```
>🔺Pour le 💲cript "LoRA"


# E - Docker Compose . YAML
```yaml
services:

  extract:
    build:
      context: .
      dockerfile: Dockerfile.extract
    volumes:
      - .:/app
    command:
      [
        "--db",
        "data/support_it.db",
        "--output",
        "extracted.jsonl"
      ]

  train:
    build:
      context: .
      dockerfile: Dockerfile.train
    volumes:
      - .:/app
    command:
      [
        "--data",
        "gold_data.jsonl"
      ]

# MlFlow part
  mlflow:
    image: ghcr.io/mlflow/mlflow
    ports:
      - "5000:5000"

```

# F - Lancement
### Construction :
```Shell
docker compose build
```

# J - Lancement
### Extraction :
```Go
docker compose run extract
```
### - Fine-tuning :
```Go
docker compose run train
```

# H - Et ...pour *MLflow* ?
Ajout d'un *3ème* service
```Go
mlflow/
```
### & dans docker-compose.yml :
> déja ajouté + haut en section E
```python
mlflow:
  image: ghcr.io/mlflow/mlflow
  ports:
    - "5000:5000"
```
### Puis dans train.py :

```Python
import mlflow

mlflow.set_experiment("support_it_lora")

with mlflow.start_run():
    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("lr", args.lr)

    trainer.train()

    mlflow.log_artifacts(str(args.output))

```

# I  Version "niveau ML Engineer"
```GO
Container 1
    |
    v
get_from_database.py
    |
    v
Container 2
transform_to_gold.py
    |
    v
Container 3
train.py
    |
    v
Container 4
serve.py (API FastAPI)
```
> Cette approche s'aligne avec les pipelines MLOps/Kubeflow déjà utilisé dans mes procédents projet d' Architecture_CLOUD, où chaque étape du workflow ML est exécutée indépendamment et orchestrée sous forme de pipeline.

## Explication :
```GO
Isolation des traitements
+
Reproductibilité
+
Portabilité
+
Préparation MLOps / Kubeflow
```
> Montre une démarche d'industrialisation du pipeline de fine-tuning.
