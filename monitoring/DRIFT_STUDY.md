# Étude de dérive des données (data drift) et surveillance opérationnelle

Partie monitoring du projet : stockage des données de
production, analyse automatique (drift + anomalies opérationnelles) et
visualisation.

## 1. Données de production collectées

| Donnée | Où / comment | Stockage |
|---|---|---|
| Inputs complets (22 colonnes brutes) | chaque `/predict` logge le payload tel que reçu en artefact MLflow `input.json` | MLflow (SQLite local, volume Docker `api-tracking` en local) |
| Outputs (prediction, probability, risk_category) | métriques et tags du run MLflow | MLflow |
| Temps d'exécution (`processing_time_ms`) | métrique du run MLflow | MLflow |
| Log de requêtes (une ligne JSON par requête : timestamp, méthode, chemin, statut, durée) | middleware HTTP de l'API | JSONL `tracking/api_requests.jsonl` (`REQUESTS_LOG_FILE`) |
| Métriques live (buffer 500 prédictions : latence avg/p95, distribution des scores, catégories de risque) | `GET /monitoring` | mémoire (redémarrage = reset) |
| Taux d'erreur (requêtes, 4xx, 5xx) | middleware HTTP comptant chaque requête, exposé sur `GET /monitoring` ET persisté dans le log de requêtes JSONL | mémoire + JSONL |

**Logging structuré** : les logs applicatifs de l'API sont émis en JSON
(une ligne = un objet avec `timestamp`, `level`, `logger`, `message`) ;
`LOG_FORMAT=text` restaure le format texte pour le débogage local. Le log
de requêtes JSONL est la pièce exploitable par machine : taux d'erreur et
latence y sont analysables a posteriori.

**Contrainte de l'hébergement** : le plan gratuit Render n'a pas de disque
durable — le store MLflow de l'API cloud est perdu à chaque redéploiement.
Comme l'énoncé l'autorise explicitement (« les logs peuvent être générés par
l'API (cloud) puis téléchargés, stockés et analysés localement »), le PoC
fonctionne ainsi :

1. le trafic passe par l'API (localement via la stack Docker dont le volume
   `api-tracking` persiste, ou contre l'API déployée avec
   `monitoring/generate_traffic.py --url ... --save-payloads ...`) ;
2. l'analyse tourne localement sur ces données.

## 2. Méthodologie d'analyse

- **Référence** : `application_train` restreint aux 22 colonnes du modèle
  (20 000 lignes échantillonnées) — la même référence que celle utilisée à
  l'entraînement (`dataset v2_form_top22`).
- **Production** : les artefacts `input.json` des runs MLflow de l'API, ou un
  export JSONL des payloads (`generate_traffic.py --save-payloads`).
- **Imputation cohérente** : les champs omis/nuls sont imputés avec la
  médiane/mode d'entraînement (celles du schéma), exactement comme le fait
  l'API — la comparaison reflète ce que le modèle a réellement vu.
- **Tests statistiques** (Evidently 0.7) : K-S par colonne numérique, Z-test
  par colonne catégorielle. Seuil **p < 0,01**, corrigé pour les 22
  comparaisons simultanées (voir §4).
- **Verdicts** :
  - *stable* : aucune colonne dérivée ;
  - *drift_detected* : ≥ 50 % des colonnes dérivées (règle dataset d'Evidently) ;
  - *alert_critical_features* : `EXT_SOURCE_1/2/3` ou `DAYS_BIRTH` dérivées —
    ces colonnes portent ~99 % du gain du modèle, elles déclenchent une alerte
    même si la part de colonnes dérivées reste sous les 50 % ;
  - *insufficient_samples* : moins de 20 prédictions, tests non exécutés.
- **Anomalies opérationnelles** : p95 de latence > 500 ms (l'API sert en
  ~10-15 ms) ; taux d'erreur > 5 % des requêtes (4xx inclus : un pic
  d'erreurs de validation signale généralement une intégration cliente
  cassée) analysé depuis le log de requêtes JSONL
  (`drift_analysis.py --requests-file`, consommé aussi par l'onglet Data
  Drift du dashboard).

Outils : `monitoring/generate_traffic.py` (trafic réaliste, avec option
`--shift` pour injecter une dérive), `monitoring/drift_analysis.py`
(l'analyse), onglet **Data Drift** du dashboard Streamlit (visualisation).

## 3. Résultats du PoC

| Scénario | Source | n production | Colonnes dérivées | Verdict |
|---|---|---|---|---|
| Trafic normal | export payloads | 500 | **0/22** | stable |
| Trafic normal | MLflow (runs API) | 150 | **0/22** | stable — latence avg 13,1 ms, p95 15,1 ms, max 44,7 ms : aucune anomalie |
| Dérive injectée (`EXT_SOURCE_2` × 0,15, `AMT_INCOME_TOTAL` × 4) | export payloads | 500 | **2/22** : `EXT_SOURCE_2`, `AMT_INCOME_TOTAL` | **alert_critical_features** (`EXT_SOURCE_2` est critique) |

La dérive injectée est détectée précisément sur les deux colonnes visées, avec
un verdict d'alerte qui ne repose pas sur la règle des 50 % (2/22 = 9 %) mais
sur la criticité des features — c'est le comportement voulu.

Rapports détaillés par feature : `monitoring/output/drift_report*.html`
(régénérés par le script, non versionnés). Summaries machine-lisibles :
`monitoring/output/drift_summary*.json` (versionnés, consommés par le
dashboard).

## 4. Points de vigilance issus de l'étude

1. **Petits échantillons + comparaisons multiples** : avec 150-190 prédictions
   et un seuil naïf p < 0,05, nous avons observé 4 à 8 colonnes sur 22
   signalées à tort sur du trafic issu de la même distribution que la
   référence (22 tests × α = 5 % ≈ 1 faux positif attendu, aggravé par le
   bruit d'échantillonnage). Correction appliquée : seuil p < 0,01 **et**
   fenêtre d'analyse ≥ 500 prédictions ; à 500 échantillons le trafic normal
   passe proprement à 0/22.
2. **Masse ponctuelle d'imputation** : `EXT_SOURCE_1` est absente pour ~57 %
   des dossiers d'entraînement ; l'imputation médiane crée un pic énorme à
   0,506 qui rend les distances de distribution (Wasserstein) très sensibles
   au bruit d'échantillonnage. Les tests à p-value (K-S) sont plus stables
   ici — d'où leur choix explicite plutôt que la sélection automatique
   d'Evidently.
3. **Règle dataset 50 % trop laxiste pour ce modèle** : avec 22 colonnes,
   une dérive réelle sur 2 colonnes qui portent 99 % du gain serait
   « noyée » dans la règle des 50 %. D'où la règle d'alerte sur features
   critiques.
4. **La référence vieillit** : on compare toujours la production aux données
   d'entraînement, qui datent du jour où le modèle a été créé ; avec le temps,
   cette comparaison finit par signaler de simples évolutions normales. La
   prochaine étape : comparer plutôt à une période récente de production
   vérifiée saine, dès qu'on aura assez de recul.
5. **Stockage / coût** : MLflow SQLite local (volume Docker) suffit au PoC ;
   en production, un store persistant (PostgreSQL/MLflow server) et une
   politique de rétention (les inputs bruts grossissent linéairement avec le
   trafic) sont nécessaires.
6. **RGPD** : le dataset Home Credit est anonymisé (identifiants techniques,
   aucune donnée personnelle directe) ; les inputs stockés restent toutefois
   des dossiers de crédit — en production réelle : minimisation (ne pas
   stocker les colonnes inutiles), durée de rétention définie et accès
   restreint au store.
7. **Éphémérité du plan gratuit Render** : le buffer `/monitoring` et le store
   MLflow se vident à chaque redéploiement — les analyses doivent être
   relançées ou les données exportées avant.

## 5. Plan d'action opérationnel

- **Fréquence** : analyse hebdomadaire, ou dès qu'une fenêtre de 500
  prédictions est atteinte.
- **En cas d'alerte** : (1) vérifier le flux de données de la colonne
  concernée (fournisseur, ETL), (2) comparer avec l'onglet Monitoring du
  dashboard (le taux de défaut bouge-t-il ?), (3) si la dérive persiste deux
  fenêtres consécutives, re-entraîner (`api/train_model.py`) avec les données
  récentes ajoutées, via la pipeline CI/CD existante.
- **Métriques opérationnelles suivies en continu** : taux d'erreur (`/monitoring`
  en direct, log JSONL pour l'historique), latence p95 (> 500 ms = anomalie),
  distribution des scores prédits.

## 6. Reproduire l'analyse

```bash
# trafic réaliste -> API locale (logged dans MLflow)
uv run python monitoring/generate_traffic.py --n 500

# analyse sur les runs MLflow récents
uv run python monitoring/drift_analysis.py --since=-2h

# scénario avec dérive injectée (démo)
uv run python monitoring/generate_traffic.py --n 500 --shift --save-only \
    --save-payloads monitoring/output/payloads_shifted.jsonl
uv run python monitoring/drift_analysis.py --label shifted \
    --inputs-file monitoring/output/payloads_shifted.jsonl
```

Puis ouvrir le dashboard (`streamlit run streamlit_app/app.py`), onglet
**Data Drift**.
