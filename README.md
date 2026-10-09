# Credit Scoring - Projet MLOps

**Projet 8 OpenClassrooms : Implémentation complète API + Docker + CI/CD + MLflow + Streamlit**

---

## 🚀 Commandes du projet

Prérequis : [uv](https://docs.astral.sh/uv/) installé. Toutes les commandes s'exécutent depuis la racine du projet.

### Installation des dépendances

```bash
uv sync                  # dépendances runtime
uv sync --all-extras     # + dépendances dev (pytest, mypy, stubs...)
```

### Entraîner le modèle

```bash
uv run python -m api.train_model
```

Cela régénère `api/models/credit_scoring_model.pkl` (modèle + preprocessor + métadonnées), `api/schema/raw_columns.json`, et crée un run MLflow dans `mlflow.db`.

Paramètres disponibles (`api/train_model.py`) :

| Paramètre | Défaut | Description |
|---|---|---|
| `--data-dir` | `data/raw` | Dossier contenant les CSV Home Credit |
| `--nrows` | toutes les lignes | Limite le nombre de lignes lues (test rapide) |
| `--model-type` | `lightgbm` | `lightgbm`, `logistic_regression` ou `random_forest` |
| `--tune` | désactivé | Active l'optimisation d'hyperparamètres Optuna |
| `--n-trials` | `10` | Nombre d'essais Optuna (avec `--tune`) |
| `--n-folds` | `3` | Nombre de folds de la validation croisée |
| `--experiment-name` | `credit_scoring_api` | Nom de l'expérience MLflow |
| `--run-name` | auto | Nom du run MLflow |
| `--tracking-uri` | `sqlite:///tracking/mlflow.db` | URI du tracking MLflow |
| `--register` | désactivé | Inscrit le modèle au MLflow Model Registry |
| `--output-path` | `api/models/credit_scoring_model.pkl` | Chemin du pickle sauvegardé |

Exemples :

```bash
# Entraînement rapide (comme la CI, ~1 min)
uv run python -m api.train_model --nrows 20000

# Entraînement complet avec tuning Optuna et inscription au registry
uv run python -m api.train_model --tune --n-trials 50 --register

# Autre type de modèle
uv run python -m api.train_model --model-type logistic_regression
```

### Lancer l'API

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- Swagger : http://localhost:8000/docs
- Health check : http://localhost:8000/health
- Schéma d'entrée : http://localhost:8000/schema

### Lancer le dashboard Streamlit

```bash
uv run streamlit run streamlit_app/app.py --server.port=8501
```

Dashboard : http://localhost:8501 (nécessite l'API démarrée, `API_URL` par défaut : `http://localhost:8000`)

### Lancer les tests

```bash
uv run pytest tests/ -v --cov=api.main
```

### Docker

```bash
# Builder les images (après l'entraînement : le pickle est copié dans l'image)
docker build -t credit-scoring-api .
docker build -t credit-scoring-api-streamlit -f Dockerfile.streamlit .

# Stack complet en local (API + dashboard)
docker-compose up -d --build
docker-compose logs -f
docker-compose down
```

### UI MLflow

```bash
uv run mlflow ui --backend-store-uri sqlite:///tracking/mlflow.db
```

Interface : http://localhost:5000

---

## 🎯 Accès à MLflow et Visualisation des Logs

### 1. Démarrer l'UI MLflow

```bash
# Démarrer l'interface web MLflow
mlflow ui --backend-store-uri sqlite:///tracking/mlflow.db
```

**Accès :** Ouvrir `http://localhost:5000` dans votre navigateur

### 2. Observer les Runs d'Entraînement

Tous les entraînements sont loggés dans l'expérience **`credit_scoring_api`** par défaut.

Dans l'UI MLflow (`http://localhost:5000`) :
1. Sélectionnez l'expérience **`credit_scoring_api`**
2. Cliquez sur un run pour voir les détails
3. Explorez les onglets :
   - **Overview** : Métriques, params, tags résumés
   - **Metrics** : courbes de cv_train_auc, cv_valid_auc, best_f1, etc.
   - **Params** : hyperparamètres du modèle, best_threshold, etc.
   - **Artifacts** : feature_importances, metrics, submissions
   - **Model** : modèle entraîné sauvegardé

### 3. Observer les Prédictions API

Toutes les prédictions via l'API sont loggées dans MLflow avec :

- **Run Name** : `api_prediction_YYYYMMDD_HHMMSS...`
- **Tags** : `source=api`, `predictor=credit_scoring`, `model_version`, `risk_category`
- **Params** : Toutes les features d'entrée (age, gender, income, etc.)
- **Metrics** : prediction (0/1), probability, processing_time_ms

Pour voir les prédictions dans l'UI :
1. Filtrez par tag `source=api`
2. Ou cherchez les runs commençant par `api_prediction_`

### 4. Accéder via Python

```python
import mlflow

# Configurer le backend
mlflow.set_tracking_uri("sqlite:///tracking/mlflow.db")

# Lister les expériences
client = mlflow.tracking.MlflowClient()
experiments = client.list_experiments()

# Lister les runs d'une expérience
exp = [e for e in experiments if e.name == "credit_scoring_api"][0]
runs = client.search_runs(experiment_ids=[exp.experiment_id])

# Récupérer les métriques d'un run
run_id = runs[0].info.run_id
metrics = client.get_run(run_id).data.metrics
print(f"CV AUC: {metrics['cv_valid_auc']}")

# Charger un modèle pour inférence
model_uri = f"runs:/{run_id}/model"
model = mlflow.pyfunc.load_model(model_uri)
```

### 5. Accéder aux Artifacts

Les artifacts sont sauvegardés dans le dossier `tracking/mlruns/` :

```
tracking/mlruns/
└── <experiment-id>/
    └── <run-id>/
        ├── artifacts/
        │   ├── feature_importances/
        │   │   └── feature_importances.csv
        │   ├── metrics/
        │   │   └── metrics.csv
        │   ├── optuna/
        │   │   └── optuna_trials.csv
        │   └── submission/
        │       ├── submission_class.csv
        │       └── submission_proba.csv
        └── model/
            └── MLmodel
```

Pour accéder directement :
```bash
# Voir la structure
ls tracking/mlruns/<experiment-id>/<run-id>/artifacts/

# Voir les métriques
cat tracking/mlruns/<experiment-id>/<run-id>/artifacts/metrics/metrics.csv

# Voir les feature importances
cat tracking/mlruns/<experiment-id>/<run-id>/artifacts/feature_importances/feature_importances.csv
```

---

## Structure Finalisée

```
.
├── api/
│   ├── __init__.py
│   ├── main.py                    # FastAPI avec MLflow tracking
│   ├── ml_pipeline.py            # Pipeline ML (feature engineering, preprocessor, training)
│   ├── train_model.py             # Entraînement avec MLflow
│   └── models/
│       └── credit_scoring_model.pkl
├── streamlit_app/
│   └── app.py                     # Dashboard Streamlit
├── tests/
│   └── test_api.py                # 15+ tests
├── data/raw/                     # Données (gardées dans repo)
├── tracking/                      # MLflow (db + artifacts) et Optuna — ignoré par Git, régénéré auto
├── .dockerignore
├── .github/workflows/ci-cd.yml
├── Dockerfile, Dockerfile.streamlit
├── docker-compose.yml
├── pyproject.toml, uv.lock
└── README.md
```
