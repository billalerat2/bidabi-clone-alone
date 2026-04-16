# BIDABI : Clone → Adapt → Create

## Description du projet

Dépôt pédagogique du cours **Big Data and Business Intelligence (BIDABI)**.
Ce projet met en œuvre un pipeline complet de **classification d'images alimentaires** à partir d'un dataset collecté sur **OpenFoodFacts** : scraping, versionnage des données et du modèle avec **DVC**, fine‑tuning d'un **ResNet‑18** (full FT + MixUp) et évaluation (matrice de confusion, ROC, t‑SNE…).
L'objectif pédagogique : reproduire un cycle réel d'ingénieur ML — cloner, adapter un projet open‑source, créer son propre jeu de données et l'intégrer dans un pipeline reproductible.

## Structure du dépôt

```
bidabi-clone-alone/
├── data/
│   ├── raw/                    # Dataset brut (versionné DVC)
│   │   ├── images/             # Images organisées par classe
│   │   │   ├── breads/
│   │   │   ├── chocolates/
│   │   │   └── sugar/
│   │   └── metadata_*.csv      # Métadonnées OpenFoodFacts
│   ├── raw.dvc                 # Pointeur DVC du dataset
│   └── localstore/             # Remote DVC local
├── models/
│   └── best_model_resnet18_finetuned.pth   # Modèle (versionné DVC)
├── models.dvc                  # Pointeur DVC du modèle
├── reports/                    # Figures d'évaluation (loss, ROC, t‑SNE…)
├── src/
│   ├── asyscrapper.py          # Scraper OpenFoodFacts
│   ├── data_loader.py          # Utilitaires de chargement
│   └── classificator.py        # Pipeline d'entraînement & évaluation
├── requirements.txt
├── LICENSE
└── README.md
```

## Pipeline

1. **Scraping** ([src/asyscrapper.py](src/asyscrapper.py)) — téléchargement d'images OpenFoodFacts par catégorie.
2. **Entraînement** ([src/classificator.py](src/classificator.py)) — ResNet‑18 pré‑entraîné, full fine‑tuning + MixUp, split 60/20/20.
3. **Évaluation** — matrice de confusion, ROC, t‑SNE, accuracy par classe → figures dans [reports/](reports/).

## Dataset

Versionné par DVC via [data/raw.dvc](data/raw.dvc).

- **Classes** : `breads`, `chocolates`, `sugar`
- **Images** : 366 (~180 / classe)
- **Hash** : `abe7f674721bab3739a3c2b5adfa2388`

## Modèle

Versionné par DVC via [models.dvc](models.dvc).

- **Fichier** : `models/best_model_resnet18_finetuned.pth`
- **Architecture** : ResNet‑18 + `Dropout(0.4) → Linear`
- **Hash** : `f73138e2b96e95ffd76f6e1e0b5832dc`

## Prérequis

- **Python 3.11+**
- **Git** + **DVC 3.x**

Dépendances ([requirements.txt](requirements.txt)) :

```
torch==2.2.2
torchvision==0.17.2
dvc==3.67.1
scikit_learn==1.8.0
numpy==1.26.4
matplotlib==3.10.8
seaborn==0.13.2
Requests==2.33.1
aiohttp==3.13.5
urllib3==2.6.3
```

## Reproduire l'entraînement

```bash
# 1. Cloner
git clone <url-du-depot>
cd bidabi-clone-alone

# 2. Environnement
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt

# 3. Récupérer dataset + modèle via DVC
dvc pull

# 4. Lancer l'entraînement
python src/classificator.py
```

Le script entraîne ResNet‑18, sauvegarde le meilleur modèle dans `best_model_resnet18_finetuned.pth` et génère les figures dans `reports/`.

Hyperparamètres par défaut : `image=256×256`, `batch=32`, `epochs=7`, `lr=1e-5`, `seed=42`.

## Licence

Dépôt pédagogique — voir [LICENSE](LICENSE).
