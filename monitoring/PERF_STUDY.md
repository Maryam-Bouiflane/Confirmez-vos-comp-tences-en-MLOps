# Étude d'optimisation du temps d'inférence post-déploiement

Analyse des performances réelles en production, identification des goulots
d'étranglement, stratégies d'optimisation testées et résultats. Tous les
chiffres ci-dessous sont mesurés sur le modèle déployé (LightGBM + CreditPreprocessor,
`api/models/credit_scoring_model.pkl`) — script de reproduction : `monitoring/profile_inference.py`,
benchmarks : `monitoring/optimize_model.py`. Résultats bruts versionnés dans
`monitoring/output/profile_results_*.json` et `monitoring/output/optimization_results.json`.

## 1. Hypothèses fondées sur les données de monitoring réelles

Le monitoring en place (`tracking/api_requests.jsonl`, MLflow, `/monitoring`) a
fourni les signaux de départ :

- première requête après démarrage : **~1 240 ms** (log de requêtes en production),
  puis régime établi à ~10-15 ms de `processing_time_ms` ;
- écart massif entre la durée totale de requête mesurée au middleware et le
  `processing_time_ms` (inférence) auto-rapporté ;
- latence p95 sous le seuil d'alerte (500 ms) mais moyenne élevée en régime établi.

Conclusion préliminaire : le goulot n'est pas le modèle, mais tout ce qui
l'entoure. Le profiling (cProfile, mesures par phase) le confirme.

## 2. Profiling : goulots d'étranglement identifiés

`monitoring/profile_inference.py` mesure deux niveaux : le chemin modèle
(construction du DataFrame → preprocessor → inférence) et la requête complète
(via l'API in-process, MLflow actif). 200 lignes réelles de `application_train`.

### 2.1 Chemin modèle (baseline, régime établi)

| Phase | Mean | p95 | Part |
|---|---|---|---|
| Construction du DataFrame (imputation) | 0,9 ms | 1,3 ms | 4,0 % |
| **Preprocessor (CreditPreprocessor.transform)** | **13,2 ms** | **20,3 ms** | **59,7 %** |
| `predict_proba` (LightGBM) | 3,9 ms | 5,1 ms | 17,5 % |
| `predict` (LightGBM, **2e passe**) | 4,2 ms | 5,5 ms | 18,7 % |
| **Total** | **22,1 ms** | | 100 % |

Goulots : (1) le preprocessor sklearn custom (polynomial features, features
domaine, encodages) ; (2) la double inférence — l'endpoint appelait
`predict_proba` **puis** `predict`, soit deux passes complètes dans le booster.

### 2.2 Requête complète (baseline, régime établi)

| Mesure | Valeur |
|---|---|
| Première requête (à froid) | **1 983 ms** (1 241 ms observé en prod) |
| Durée totale moyenne (hors 1re) | **332,5 ms** (p95 : 367,9 ms) |
| `processing_time_ms` moyen | 24,4 ms |
| Overhead de serving | **308,1 ms** |

Mesure isolée (`log_prediction_to_mlflow`) : **~255 ms par prédiction** —
83 % de l'overhead vient du logging MLflow **synchrone** (un run sqlite complet :
params + artefact `input.json` + métriques, transaction par prédiction).

## 3. Stratégies d'optimisation testées

### O1 — Warm-up au démarrage (`api/main.py:warm_up_inference`)

Une prédiction complète est exécutée sur l'événement de startup : le coût de
démarrage (imports, première passe LightGBM, session ONNX) est payé au
redéploiement, pas au premier client.

Résultat : première requête **1 983 ms → 49 ms**.

### O2 — Inférence single-pass (`api/main.py:predict_probability`)

La classe est dérivée de la probabilité (`p > 0,5`, règle de décision propre
de LightGBM) au lieu d'un second appel `model.predict`. Vérification de
parité sur lignes réelles : 0 divergence (voir §4).

Résultat : la moitié du coût d'inférence LightGBM disparaît (deux passes → une).

### O3 — Logging MLflow asynchrone (`MLFLOW_LOG_MODE=async|sync|off`)

Le run MLflow par prédiction est soumis à un `ThreadPoolExecutor`
(1 worker : sqlite n'accepte pas les écritures concurrentes) au lieu de
bloquer la requête. Les threads non-daemon sont joints à l'arrêt du
processus : les outils batch (`generate_traffic.py`) ne perdent aucun run.
`MLFLOW_LOG_MODE=sync` restaure le comportement d'origine, `off` le
désactive en production (le log JSONL de requêtes et `/monitoring` restent).

Résultat : overhead de serving **308 ms → 6,6 ms** par requête.

### O4 — Moteur d'inférence ONNX Runtime (`INFERENCE_ENGINE=auto|onnx|lightgbm`)

Le LGBMClassifier est exporté en ONNX (`monitoring/optimize_model.py`,
onnxmltools, zipmap désactivé) → `api/models/credit_scoring_model.onnx`
(22 Mo, committé, déployé avec l'image). Le preprocessor custom reste en
sklearn : seul le booster est exporté (limite documentée, §6).
Fallback automatique et silencieux vers LightGBM si l'artefact ou
`onnxruntime` est absent ; vérification de largeur d'entrée au chargement
(un pickle et un ONNX désynchronisés retombent en LightGBM plutôt que de
prédire faux).

Benchmark single-row (300 lignes réelles, après warm-up, `optimization_results.json`) :

| Moteur | mean | p50 | p95 |
|---|---|---|---|
| LightGBM (`predict_proba`) | 4,27 ms | 2,88 ms | 18,07 ms |
| **ONNX Runtime** | **0,39 ms** | **0,14 ms** | **0,29 ms** |

**x11 en moyenne, x60 sur la p95.** Batch(32) à titre de perspective :
37,1 ms → 9,4 ms.

## 4. Non-régression (précision, biais)

Comparaison systématique des probabilités LightGBM vs ONNX sur 300 lignes
réelles (`optimize_model.py`, rejouable) :

- écart maximal de probabilité : **2,1e-7** ;
- flips de décision au seuil 0,5 : **0 / 300** ;
- flips au seuil métier (0,668) : **0 / 300**.

La règle de décision (probabilité > 0,5) est celle de LightGBM lui-même —
testée par `tests/test_api.py::TestSinglePassPrediction`. Un test de parité
ONNX/LightGBM sur données synthétiques (`TestOnnxEngine`) est skippé
proprement si `onnxruntime` ou l'artefact est absent (CI sans l'extra `onnx`).
Le preprocessor, source de features identiques, est inchangé → pas de
risque de biais supplémentaire : mêmes features, mêmes arbres, mêmes
prédictions à 2e-7 près.

## 5. Résultat global et déploiement

Requête complète en régime établi (mêmes 200 lignes, même protocole) :

| | Baseline | Optimisée | Gain |
|---|---|---|---|
| Première requête | 1 983 ms | 49 ms | x40 |
| Requête moyenne | 332,5 ms | 27,4 ms | **x12** |
| p95 | 367,9 ms | 38,9 ms | x9,5 |
| Overhead de serving | 308,1 ms | 6,6 ms | x47 |

La version optimisée est intégrée au dépôt (`.onnx` committé, `INFERENCE_ENGINE`
et `MLFLOW_LOG_MODE` documentés) et déployée par la pipeline CI/CD existante :
le Dockerfile installe l'extra `onnx` (`uv sync --frozen --extra onnx`), le
startup charge la session ONNX et exécute le warm-up. `GET /model` expose le
moteur actif (`inference_engine: onnx|lightgbm`).

## 6. Justification de la configuration finale

- **Librairies** : FastAPI/uvicorn (inchangé, l'overhead HTTP est négligeable
  devant les gains mesurés) ; LightGBM conservé pour l'entraînement (accuracy
  au treining inchangée) ; **ONNX Runtime 1.31** pour l'inférence (CPU,
  single-thread suffit à ~0,4 ms) ; sklearn inchangé pour le preprocessor.
  Pas de quantification : le booster LightGBM est déjà en feuilles à seuils —
  la quantification n'aurait pas de gain mesurable ici et introduirait un
  risque de régression inutile (l'énoncé la propose comme piste, pas comme
  obligation ; le benchmark a préféré ONNX qui est exact au float32 près).
- **Software** : Python 3.12, image `python:3.12-slim` + `libgomp1` (OpenMP
  LightGBM), logging MLflow asynchrone par défaut (`sync`/`off` disponibles).
- **Hardware** : CPU only — à 0,4 ms d'inférence ONNX, un GPU serait
  inutile (coût/complexité sans bénéfice à ce débit). Le plan Render
  gratuit est **limité à 512 Mo de RAM** : maintenir en mémoire le booster
  LightGBM dé-picklé ET la session ONNX dépasse cette limite (OOM observé
  au démarrage du staging). Décision : `INFERENCE_ENGINE=lightgbm` sur
  Render (blueprints), moteur ONNX actif localement et sur tout
  environnement disposant de RAM suffisante — les autres optimisations
  (logging asynchrone = ~80 % du gain, warm-up, single-pass) sont
  neutres en mémoire et restent actives en production. C'est l'illustration
  du point de vigilance de l'énoncé : valider la compatibilité des
  optimisations avec l'environnement de production réel.

## 7. Limites et perspectives

- Le preprocessor (13-27 ms selon la charge machine) est désormais le
  principal poste du chemin modèle : vectorisation/parallélisation de
  `CreditPreprocessor` ou export complet du pipeline en ONNX = prochaine
  étape naturelle.
- Les mesures sont faites en local (API in-process, TestClient) : les
  proportions se retrouvent en production (mêmes composants), pas les
  valeurs absolues (Render free tier = CPU partagé).
- Les timings varient avec la charge de la machine de mesure (les chiffres
  des tableaux proviennent d'un même run par scénario ; les JSON bruts
  versionnés font foi).

## 8. Reproduire

```bash
# 1. Baseline (code pré-optimisation : voir profile_results_baseline.json)
uv run python monitoring/profile_inference.py --n 200 --label baseline

# 2. Optimisations + mesures
uv run --extra onnx python monitoring/optimize_model.py   # conversion + parité + benchmark
uv run --extra onnx python monitoring/profile_inference.py --n 200 --label optimized

# 3. Vérification de non-régression (tests)
uv run --extra onnx pytest tests/ -q
```
