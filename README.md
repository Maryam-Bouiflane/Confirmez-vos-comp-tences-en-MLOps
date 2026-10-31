# Credit Scoring - Projet MLOps

**Projet 8 OpenClassrooms : Implémentation complète API + Docker + CI/CD + MLflow + Streamlit**

---

## 🚀 Quick Start

### Lancer en local (5 minutes)

```bash
# 1. Prérequis : Python 3.12+ et uv (https://docs.astral.sh/uv/)
# 2. Cloner puis se placer à la racine du projet
# 3. Installer les dépendances
uv sync

# 4. Lancer l'API (le modèle entraîné est committé : rien à entraîner)
uv run uvicorn api.main:app --port 8000
#    -> Swagger : http://localhost:8000/docs

# 5. Dans un second terminal, lancer le dashboard
uv run streamlit run streamlit_app/app.py --server.port=8501
#    -> http://localhost:8501
```

Remarques :

- Les CSV de `data/raw/` ne sont **pas nécessaires** pour lancer l'API et le dashboard (le modèle est déjà entraîné) — ils ne servent qu'à ré-entraîner ou rejouer les analyses de monitoring (voir « Données requises »).
- Le premier appel à `/predict` peut prendre quelques secondes de plus (warm-up du moteur, payé au démarrage depuis l'optimisation).

### Consulter la version déployée (Render)

L'API et le dashboard sont déployés automatiquement par la pipeline CI/CD (branche `main`, déploiement après succès des checks GitHub Actions) :

- **API** : https://credit-scoring-api-onnj.onrender.com — Swagger sur `/docs`, état de santé sur `/health`, moteur d'inférence actif sur `/model`
- **Dashboard** : service Render `credit-scoring-dashboard` (URL visible dans la console Render) — l'URL de l'API est préconfigurée
- **Staging** : services `credit-scoring-api-staging` / `credit-scoring-dashboard-staging`, déployés depuis la branche `develop` (`render.staging.yaml`)

Attention (plan Render gratuit) : après une période d'inactivité, le service se met en veille — la première requête peut prendre ~1 minute. Réveiller l'API en visitant d'abord `/health`, puis ouvrir le dashboard.

---

## 🛠️ Commandes du projet

Prérequis : [uv](https://docs.astral.sh/uv/) installé. Toutes les commandes s'exécutent depuis la racine du projet.

### Installation des dépendances

```bash
uv sync                  # dépendances runtime
uv sync --all-extras     # + dépendances dev (pytest, mypy, stubs...)
```

### Données requises : placer les CSV dans `data/raw/`

`data/raw/` est ignoré par Git (fichiers volumineux). **Les deux CSV ci-dessous sont fournis dans le zip du projet** : les placer dans `data/raw/` pour que le projet fonctionne.

| CSV | Obligatoire pour |
|---|---|
| `application_train.csv` | **Tout** : entraînement du modèle, génération de trafic, analyse de drift (données de référence), benchmarks d'optimisation |
| `application_test.csv` | L'entraînement complet uniquement (génération des artefacts de soumission dans MLflow) |

À noter : sans ces CSV, l'API fonctionne quand même (le modèle entraîné est committé dans `api/models/credit_scoring_model.pkl`), mais l'entraînement (`api/train_model.py`) et les analyses de monitoring (`drift_analysis.py`, `generate_traffic.py`, benchmarks) ne peuvent pas être relancés.

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

## 📊 Monitoring en production : drift + anomalies opérationnelles

La partie monitoring est décrite en détail dans [`monitoring/DRIFT_STUDY.md`](monitoring/DRIFT_STUDY.md) et le support de présentation est [`monitoring/presentation_drift.pptx`](monitoring/presentation_drift.pptx).

### Données collectées en production

- **Inputs** : chaque `/predict` logge le payload complet en artefact MLflow `input.json`
- **Outputs** : `prediction`, `probability`, `risk_category` (metrics/tags MLflow)
- **Temps d'exécution** : `processing_time_ms` (metric MLflow)
- **Log de requêtes** : le middleware HTTP écrit une ligne JSON par requête (timestamp, méthode, chemin, statut, durée) dans `tracking/api_requests.jsonl` — persiste le taux d'erreur et la latence pour analyse ultérieure (`REQUESTS_LOG_FILE` pour changer le chemin)
- **Logging structuré** : les logs applicatifs sont émis en JSON (une ligne = un objet) ; `LOG_FORMAT=text` restaure le format texte
- **Métriques live** : `GET /monitoring` (buffer des 500 dernières prédictions, latence avg/p95, taux d'erreur en direct)

### Stockage des données de production : solution en place et cible en production

**En place (PoC)** — conforme à l'énoncé (« logs générés par l'API puis stockés et analysés localement ») :

| Donnée | Stockage actuel |
|---|---|
| Inputs complets, outputs, temps d'exécution | MLflow : un run par prédiction (artefact `input.json`, metrics, tags) — SQLite local, volume Docker `api-tracking` |
| Statut et durée de chaque requête | `tracking/api_requests.jsonl` (une ligne JSON par requête, écrit par le middleware) |
| Métriques live (latence, taux d'erreur) | Buffer mémoire servi par `GET /monitoring` |
| Logs applicatifs | JSON sur la sortie standard (une ligne = un objet) |

Limite assumée du PoC : MLflow est un outil de tracking d'expérimentation, pas un entrepôt de logs ; le plan Render gratuit n'a pas de disque durable (les données sont exportées avant redéploiement, l'analyse tourne en local).

**Cible en production réelle** :

1. **Logs applicatifs** : le JSON sur stdout est collecté par un agent **Fluentd** (ou Logstash) et indexé dans **Elasticsearch**, visualisable dans Kibana — notre logging structuré (1 ligne = 1 objet JSON) est directement exploitable, sans parsing.
2. **Journal des prédictions** : les inputs/outputs/temps d'exécution vont dans une table **PostgreSQL** `predictions` (timestamp, model_version, les 22 inputs, probability, prediction, risk_category, processing_time_ms) — requêtable, avec partitions mensuelles et politique de rétention ; c'est cette table qui alimenterait l'analyse de drift.
3. **Métriques opérationnelles** : latence p95 et taux d'erreur exposés pour **Prometheus** (scrape de `/monitoring`), dashboards et alertes **Grafana** (p95 > 500 ms, taux d'erreur > 5 %).
4. **Tracking d'entraînement** : MLflow en serveur avec backend **PostgreSQL** (persistant), et non plus SQLite local.
5. **RGPD** : minimisation (ne stocker que les 22 colonnes utiles), rétention définie (ex. 12 mois), accès restreint au store.

### Analyse automatique (drift + anomalies)

```bash
# 1. Générer du trafic réaliste vers l'API (logged dans MLflow)
uv run python monitoring/generate_traffic.py --n 500

# 2. Lancer l'analyse : drift (Evidently, K-S / Z-test, p < 0.01),
#    latence p95 (> 500 ms = anomalie) et taux d'erreur (> 5% = anomalie)
uv run python monitoring/drift_analysis.py --since=-2h

# Variante A : trafic normal exporté en fichier (données de production portables)
uv run python monitoring/generate_traffic.py --n 500 --save-only \
    --save-payloads monitoring/output/payloads_normal.jsonl
uv run python monitoring/drift_analysis.py \
    --inputs-file monitoring/output/payloads_normal.jsonl

# Variante B : scénario avec dérive injectée (démo)
uv run python monitoring/generate_traffic.py --n 500 --shift --save-only \
    --save-payloads monitoring/output/payloads_shifted.jsonl
uv run python monitoring/drift_analysis.py --label shifted \
    --inputs-file monitoring/output/payloads_shifted.jsonl
```

Sorties : `monitoring/output/drift_summary*.json` (versionnés, consommés par le dashboard) et `drift_report*.html` (rapports Evidently par feature). Le verdict combine dérive par colonne, alerte sur features critiques (`EXT_SOURCE_1/2/3`, `DAYS_BIRTH`), latence p95 et taux d'erreur.

### Paramètres de `generate_traffic.py`

| Paramètre | Défaut | Rôle |
|---|---|---|
| `--n` | 100 | Nombre de demandes (lignes réelles de `application_train.csv`) envoyées à l'API |
| `--source` | `data/raw/application_train.csv` | CSV d'où sont tirées les lignes |
| `--url` | in-process | API cible. Par défaut : API interne (TestClient, prédictions loggées dans le MLflow local, aucun serveur requis) ; sinon URL d'une API lancée (uvicorn, docker compose, Render) |
| `--shift` | off | Injecte une dérive simulée : `EXT_SOURCE_2` x0,15 et `AMT_INCOME_TOTAL` x4 |
| `--save-payloads <fichier>` | — | Exporte ce qui a été envoyé en JSONL (une ligne par demande) : sauvegarde portable des inputs de production, analysable avec `--inputs-file` |
| `--save-only` | off | Écrit le fichier de payloads **sans rien envoyer** (scénario de test sans polluer le store MLflow) |
| `--seed` | 42 | Graine aléatoire (échantillonnage reproductible) |

### Paramètres de `drift_analysis.py`

| Paramètre | Défaut | Rôle |
|---|---|---|
| `--inputs-file` | — | Source de production : fichier JSONL de payloads (alternative aux runs MLflow) |
| `--label` | baseline | Nom de l'analyse : suffixe des fichiers de sortie (`baseline` = sans suffixe) et entrée du menu dans le dashboard |
| `--since` / `--until` | — | Filtre temporel des runs MLflow (ISO ou relatif `-6h`) — exclut les vieux runs de test du store local |
| `--requests-file` | `tracking/api_requests.jsonl` | Log de requêtes de l'API : analyse du taux d'erreur (> 5 % = anomalie). Ignoré si absent ; `''` pour désactiver |
| `--reference` | `data/raw/application_train.csv` | Données de référence (entraînement) |
| `--max-reference-rows` | 20000 | Taille de l'échantillon de référence |
| `--min-samples` | 20 | Moins de prédictions que ça : verdict `insufficient_samples` (tests non exécutés) |
| `--out-dir` | `monitoring/output` | Dossier des sorties (JSON + HTML) |
| `--tracking-uri`, `--experiment`, `--schema` | — | Advanced : store MLflow à lire, expérience, schéma des 22 colonnes |

### Visualisation

Ouvrir le dashboard Streamlit (ci-dessus), onglet **Data Drift** (résultats de l'analyse) et **Live Monitoring** (métriques temps réel de `/monitoring`).

---

## ⚡ Optimisation du temps d'inférence post-déploiement

L'étude complète (profiling, goulots identifiés, optimisations, non-régression) est dans [`monitoring/PERF_STUDY.md`](monitoring/PERF_STUDY.md).

### Goulots identifiés (profiling cProfile + données de monitoring)

1. Logging MLflow **synchrone** : ~255 ms par prédiction (83 % de l'overhead de requête)
2. Preprocessor : 13,2 ms (60 % du chemin modèle)
3. Double inférence : `predict` + `predict_proba` = 2 passes dans le booster
4. Première requête à froid : ~1,2-2 s

### Optimisations intégrées (résultats mesurés)

| Optimisation | Effet |
|---|---|
| Warm-up au startup | 1re requête 1 983 ms → 49 ms |
| Inférence single-pass (classe dérivée de la probabilité) | 2 passes → 1 |
| Logging MLflow asynchrone (`MLFLOW_LOG_MODE=async/sync/off`) | overhead de requête 308 ms → 6,6 ms |
| Moteur ONNX Runtime (`INFERENCE_ENGINE=auto/onnx/lightgbm`, fallback LightGBM) | inférence x11 (p95 x60), parité 2e-7 |

Requête complète en régime établi : **332,5 ms → 27,4 ms (x12)**. Non-régression
vérifiée : 0 flip de décision sur 300 lignes réelles. Le modèle ONNX
(`api/models/credit_scoring_model.onnx`) est committé et déployé par la
pipeline CI/CD (le Dockerfile installe l'extra `onnx`).

### Reproduire les mesures

```bash
uv run --extra onnx python monitoring/optimize_model.py    # conversion ONNX + parité + benchmark
uv run --extra onnx python monitoring/profile_inference.py --n 200 --label optimized
```

Le moteur actif est exposé par `GET /model` (`inference_engine`).

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
│   ├── main.py                    # FastAPI : MLflow tracking, log de requêtes JSON, moteur ONNX/LightGBM
│   ├── ml_pipeline.py            # Pipeline ML (feature engineering, preprocessor, training)
│   ├── train_model.py             # Entraînement avec MLflow
│   └── models/
│       ├── credit_scoring_model.pkl   # modèle LightGBM + preprocessor (référence)
│       └── credit_scoring_model.onnx  # moteur d'inférence optimisé (x11, parité 2e-7)
├── monitoring/
│   ├── DRIFT_STUDY.md             # Étude de drift : méthodologie, résultats, vigilances
│   ├── PERF_STUDY.md              # Étude d'optimisation : profiling, goulots, résultats
│   ├── presentation_drift.pptx    # Support de présentation de l'étude drift
│   ├── drift_analysis.py          # Analyse drift + latence + taux d'erreur
│   ├── generate_traffic.py        # Génération de trafic réaliste (option --shift)
│   ├── profile_inference.py       # Profiling cProfile du chemin d'inférence
│   ├── optimize_model.py          # Conversion ONNX + benchmark + non-régression
│   └── output/                    # drift_summary*.json, profile/optimization results (dashboard, preuves)
├── streamlit_app/
│   └── app.py                     # Dashboard Streamlit (onglets Drift + Monitoring)
├── tests/
│   ├── test_api.py                # 45+ tests
│   └── test_drift_analysis.py     # tests drift / latence / taux d'erreur
├── data/raw/                     # CSV Home Credit — ignoré par Git : placer les 2 CSV fournis dans le zip
├── tracking/                      # MLflow (db + artifacts), Optuna, api_requests.jsonl — ignoré par Git, régénéré auto
├── .dockerignore
├── .github/workflows/ci-cd.yml
├── Dockerfile, Dockerfile.streamlit
├── docker-compose.yml
├── pyproject.toml, uv.lock
└── README.md
```
