"""
Script to train credit scoring model using ml_pipeline.py with MLflow tracking.

This script uses the existing pipeline from ml_pipeline.py to:
1. Load and preprocess the credit scoring dataset
2. Train a model (LightGBM, Logistic Regression, or Random Forest)
3. Log everything with MLflow (metrics, params, artifacts, model)
4. Save the trained model for the API

Usage:
    python api/train_model.py --model-type lightgbm --tune
    python api/train_model.py --model-type random_forest
    python api/train_model.py --experiment-name credit_scoring_api

The trained model will be saved to api/models/credit_scoring_model.pkl
"""

import argparse
import json
import os
import tempfile
import shutil
from pathlib import Path

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import mlflow.lightgbm

# Import the pipeline from ml_pipeline
from api.ml_pipeline import (
    load_application_files,
    add_polynomial_features,
    add_domain_features,
    encode_categoricals,
    train_model as pipeline_train_model,
    tune_optuna,
    apply_threshold,
    CreditPreprocessor,
    _sanitize_feature_names,
    setup_mlflow_experiment,
)


# Path of the raw schema served by the API (/schema endpoint) and used by the
# Streamlit dashboard to build its form dynamically.
SCHEMA_PATH = Path(__file__).parent / "schema" / "raw_columns.json"

# Dashboard form groups (prefix-based, French labels)
GROUP_RULES = [
    (("NAME_CONTRACT_TYPE", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "NAME_TYPE_SUITE"), "Prêt"),
    (("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"), "Sources externes"),
    (("FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"), "Contact"),
    (("FLAG_DOCUMENT_",), "Documents"),
    (("AMT_REQ_CREDIT_BUREAU_",), "Bureau de crédit"),
    (("OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE"), "Entourage"),
    (("APARTMENTS_", "BASEMENTAREA_", "YEARS_BEGINEXPLUATATION_", "YEARS_BUILD_", "COMMONAREA_",
      "ELEVATORS_", "ENTRANCES_", "FLOORSMAX_", "FLOORSMIN_", "LANDAREA_", "LIVINGAPARTMENTS_",
      "LIVINGAREA_", "NONLIVINGAPARTMENTS_", "NONLIVINGAREA_", "FONDKAPREMONT_MODE", "HOUSETYPE_MODE",
      "TOTALAREA_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE"), "Habitat"),
    (("REGION_", "REG_REGION_", "REG_CITY_", "LIVE_REGION_", "LIVE_CITY_"), "Région"),
]

# French descriptions for the most influent fields (shown in the dashboard)
DESCRIPTIONS = {
    "AMT_INCOME_TOTAL": "Revenu annuel total du client",
    "AMT_CREDIT": "Montant du crédit demandé",
    "AMT_ANNUITY": "Mensualité de l'annuité",
    "AMT_GOODS_PRICE": "Prix du bien financé",
    "DAYS_BIRTH": "Âge en jours (négatif, relatif à la demande)",
    "DAYS_EMPLOYED": "Ancienneté emploi en jours (négatif)",
    "DAYS_REGISTRATION": "Jours depuis le dernier changement d'inscription",
    "DAYS_ID_PUBLISH": "Jours depuis le dernier changement de pièce d'identité",
    "DAYS_LAST_PHONE_CHANGE": "Jours depuis le dernier changement de téléphone",
    "EXT_SOURCE_1": "Score externe normalisé 1",
    "EXT_SOURCE_2": "Score externe normalisé 2",
    "EXT_SOURCE_3": "Score externe normalisé 3",
    "CODE_GENDER": "Genre (F, M, XNA)",
    "CNT_CHILDREN": "Nombre d'enfants",
    "CNT_FAM_MEMBERS": "Nombre de membres du foyer",
    "NAME_EDUCATION_TYPE": "Niveau d'études",
    "NAME_INCOME_TYPE": "Type de revenus",
    "NAME_FAMILY_STATUS": "Situation familiale",
    "OCCUPATION_TYPE": "Profession",
    "ORGANIZATION_TYPE": "Type d'employeur",
    "OWN_CAR_AGE": "Âge du véhicule",
    "REGION_POPULATION_RELATIVE": "Population de la région (normalisée)",
    "REGION_RATING_CLIENT": "Note de la région (1-3)",
}


def get_group(column: str) -> str:
    """Map a raw column to a dashboard form group."""
    for prefixes, group in GROUP_RULES:
        for p in prefixes:
            if column == p or column.startswith(p):
                return group
    return "Client"


def generate_schema(app_train: pd.DataFrame, output_path: Path = None) -> dict:
    """Generate the raw input schema (JSON) used by the API and the dashboard.

    For each raw application column (excluding SK_ID_CURR and TARGET):
    - type: 'numeric' or 'categorical'
    - group: dashboard form group
    - default: median (numeric) or mode (categorical) computed on the
      training data; missing fields are filled with these defaults at
      serving time
    - choices: sorted unique values (categorical only)
    - description: French label when documented, else the column name
    """
    output_path = output_path or SCHEMA_PATH
    fields = []
    for col in app_train.columns:
        if col in ("SK_ID_CURR", "TARGET"):
            continue
        if app_train[col].dtype == "object":
            mode = app_train[col].mode(dropna=True)
            fields.append({
                "name": col,
                "type": "categorical",
                "group": get_group(col),
                "default": str(mode.iloc[0]) if len(mode) else "",
                "choices": sorted(app_train[col].dropna().astype(str).unique().tolist()),
                "description": DESCRIPTIONS.get(col, col),
            })
        else:
            median = app_train[col].median()
            fields.append({
                "name": col,
                "type": "numeric",
                "group": get_group(col),
                "default": round(float(median), 4) if pd.notna(median) else 0.0,
                "description": DESCRIPTIONS.get(col, col),
            })

    schema = {
        "dataset": "application_train",
        "version": "v1_application_raw",
        "n_fields": len(fields),
        "fields": fields,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    print(f"Schema written to {output_path} ({len(fields)} fields)")
    return schema


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train credit scoring model with MLflow tracking for API.'
    )
    parser.add_argument('--data-dir', default='data/raw',
                        help='Path to directory containing CSV files')
    parser.add_argument('--nrows', type=int, default=None,
                        help='Number of rows to read from CSVs (for debugging)')
    parser.add_argument('--experiment-name', default='credit_scoring_api',
                        help='MLflow experiment name')
    parser.add_argument('--run-name', default=None,
                        help='MLflow run name')
    parser.add_argument('--tracking-uri', default='sqlite:///tracking/mlflow.db',
                        help='MLflow tracking URI')
    parser.add_argument('--model-type', 
                        choices=['lightgbm', 'logistic_regression', 'random_forest'],
                        default='lightgbm',
                        help='Type of model to train')
    parser.add_argument('--tune', action='store_true',
                        help='Use Optuna for hyperparameter tuning')
    parser.add_argument('--n-trials', type=int, default=10,
                        help='Number of Optuna trials')
    parser.add_argument('--n-folds', type=int, default=3,
                        help='Number of CV folds')
    parser.add_argument('--register', action='store_true',
                        help='Register model in MLflow Model Registry')
    parser.add_argument('--output-path', default='api/models/credit_scoring_model.pkl',
                        help='Path to save the trained model')
    return parser.parse_args()


def build_run_name(model_type: str, tune: bool, n_folds: int) -> str:
    """Build MLflow run name."""
    mode = "optuna_best" if tune else "baseline"
    return f"{model_type}__{mode}__cv{n_folds}"


def log_model_with_mlflow(final_model, model_type: str, feature_names: list, 
                         best_threshold: float, best_f1: float, 
                         best_recall: float, best_precision: float,
                         metrics: pd.DataFrame, submission_proba: pd.DataFrame,
                         submission_class: pd.DataFrame, feature_importances: pd.DataFrame = None):
    """
    Log the trained model and all artifacts with MLflow.
    
    Args:
        final_model: Trained model to log
        model_type: Type of model (lightgbm, logistic_regression, random_forest)
        feature_names: List of feature names
        best_threshold: Optimal threshold for classification
        best_f1: Best F1 score
        best_recall: Best recall score
        best_precision: Best precision score
        metrics: DataFrame with CV metrics
        submission_proba: DataFrame with probability predictions
        submission_class: DataFrame with class predictions
        feature_importances: DataFrame with feature importances
    """
    # Log tags
    mlflow.set_tag("model_type", model_type)
    mlflow.set_tag("dataset", "credit_scoring")
    mlflow.set_tag("use_case", "api_deployment")
    
    # Log parameters
    mlflow.log_param("best_threshold", best_threshold)
    mlflow.log_param("n_features", len(feature_names))
    
    # Log model-specific parameters
    if hasattr(final_model, 'get_params'):
        model_params = final_model.get_params()
        for k, v in model_params.items():
            mlflow.log_param(f"model_{k}", str(v))
    
    # Log metrics
    mlflow.log_metric("best_f1", best_f1)
    mlflow.log_metric("best_recall", best_recall)
    mlflow.log_metric("best_precision", best_precision)
    
    # Log CV metrics
    for _, row in metrics.iterrows():
        fold = row["fold"]
        if fold == "overall":
            mlflow.log_metric("cv_train_auc", float(row["train"]))
            mlflow.log_metric("cv_valid_auc", float(row["valid"]))
        else:
            mlflow.log_metric(f"train_auc_fold_{int(fold)}", float(row["train"]))
            mlflow.log_metric(f"valid_auc_fold_{int(fold)}", float(row["valid"]))
    
    # Create temporary directory for artifacts
    tmp_dir = tempfile.mkdtemp()
    
    try:
        # Log feature importances
        if feature_importances is not None:
            fi_path = os.path.join(tmp_dir, "feature_importances.csv")
            feature_importances.to_csv(fi_path, index=False)
            mlflow.log_artifact(fi_path, artifact_path="feature_importances")
        
        # Log metrics
        metrics_path = os.path.join(tmp_dir, "metrics.csv")
        metrics.to_csv(metrics_path, index=False)
        mlflow.log_artifact(metrics_path, artifact_path="metrics")
        
        # Log submissions
        sub_proba_path = os.path.join(tmp_dir, "submission_proba.csv")
        submission_proba.to_csv(sub_proba_path, index=False)
        mlflow.log_artifact(sub_proba_path, artifact_path="submission")
        
        sub_class_path = os.path.join(tmp_dir, "submission_class.csv")
        submission_class.to_csv(sub_class_path, index=False)
        mlflow.log_artifact(sub_class_path, artifact_path="submission")
        
        # Log model
        if model_type == 'lightgbm':
            mlflow.lightgbm.log_model(final_model, "model")
        else:
            mlflow.sklearn.log_model(final_model, "model")
        
    finally:
        # Clean up temp directory
        shutil.rmtree(tmp_dir, ignore_errors=True)


def train_and_save():
    """Main training function."""
    args = parse_args()
    
    # Set up MLflow tracking
    setup_mlflow_experiment(args.experiment_name, args.tracking_uri)
    
    # Set run name
    run_name = args.run_name or build_run_name(
        args.model_type, args.tune, args.n_folds
    )
    
    print(f"Starting training: {run_name}")
    print(f"Model type: {args.model_type}")
    print(f"Tuning: {args.tune}")
    
    # Load data
    print("\nLoading data...")
    app_train, app_test = load_application_files(
        data_dir=args.data_dir,
        nrows=args.nrows
    )

    # Fit the inference preprocessor on the training data
    # (saved in the .pkl and reused by the API to transform raw inputs)
    print("Fitting inference preprocessor...")
    preprocessor = CreditPreprocessor().fit(app_train, app_test)

    # Regenerate the raw input schema served by the API
    generate_schema(app_train)
    
    # Feature engineering
    print("Applying feature engineering...")
    app_train_poly, app_test_poly = add_polynomial_features(app_train, app_test)
    app_train_fe, app_test_fe = add_domain_features(app_train_poly, app_test_poly)
    train_encoded, test_encoded = encode_categoricals(app_train_fe, app_test_fe)
    
    if 'SK_ID_CURR' not in train_encoded.columns or 'TARGET' not in train_encoded.columns:
        raise RuntimeError('Encoded train dataframe must contain SK_ID_CURR and TARGET')
    
    # Set dataset info
    DATASET_VERSION = 'v1_application_raw'
    FEATURES = "v1_poly_domain_encoding"
    
    # Get feature names (without target and ID), sanitized exactly as
    # _prepare_training_data does before fitting the model
    feature_names = _sanitize_feature_names(
        [col for col in train_encoded.columns
         if col not in ['SK_ID_CURR', 'TARGET']]
    )
    
    print(f"Features: {feature_names}")
    print(f"Number of features: {len(feature_names)}")
    
    best_params = None
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_name):
        # Log tags
        mlflow.set_tag("dataset_version", DATASET_VERSION)
        mlflow.set_tag("feature_set", FEATURES)
        mlflow.set_tag("tuning", str(args.tune))
        mlflow.set_tag("n_folds", str(args.n_folds))
        
        # Log basic parameters
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("n_folds", args.n_folds)
        mlflow.log_param("n_samples", len(train_encoded))
        
        # Optuna tuning
        if args.tune:
            print("\nRunning Optuna hyperparameter tuning...")
            mlflow.log_param("optuna_n_trials", args.n_trials)
            mlflow.set_tag("optuna_n_trials", str(args.n_trials))
            
            study, best_params = tune_optuna(
                train_encoded,
                n_trials=args.n_trials,
                n_folds=args.n_folds,
                model_type=args.model_type
            )
            
            # Log Optuna parameters
            if best_params:
                mlflow.log_params({f"optuna_{k}": v for k, v in best_params.items()})
            
            # Log Optuna trials
            trials_df = study.trials_dataframe()
            tmp_path = os.path.join(tempfile.gettempdir(), 'optuna_trials.csv')
            trials_df.to_csv(tmp_path, index=False)
            mlflow.log_artifact(tmp_path, artifact_path='optuna')
            os.unlink(tmp_path)
        
        # Train model
        print("\nTraining model...")
        submission, feature_importances, metrics, final_model, \
            best_recall, best_precision, best_threshold, best_f1 = pipeline_train_model(
            train_encoded,
            test_encoded,
            model_type=args.model_type,
            model_kwargs=best_params or {},
            n_folds=args.n_folds
        )
        
        submission_proba = submission
        submission_class = apply_threshold(submission, best_threshold)
        
        # Log everything with MLflow
        print("\nLogging with MLflow...")
        log_model_with_mlflow(
            final_model=final_model,
            model_type=args.model_type,
            feature_names=feature_names,
            best_threshold=best_threshold,
            best_f1=best_f1,
            best_recall=best_recall,
            best_precision=best_precision,
            metrics=metrics,
            submission_proba=submission_proba,
            submission_class=submission_class,
            feature_importances=feature_importances
        )
        
        # Log run comment
        overall_auc = metrics.loc[metrics['fold'] == 'overall', 'valid'].iloc[0]
        comment = f"""
Run MLflow - Credit Scoring API

Model: {args.model_type}
Tuning: {"Optuna" if args.tune else "Baseline"}
Cross-validation: {args.n_folds} folds
Feature pipeline: {FEATURES}
Dataset: {DATASET_VERSION}

Objective: maximize ROC-AUC

Best CV AUC valid: {overall_auc:.4f}
Best CV F1: {best_f1:.4f}
Best CV Recall: {best_recall:.4f}
Best CV Precision: {best_precision:.4f}
Best Threshold: {best_threshold:.4f}
        """
        mlflow.set_tag("mlflow.note.content", comment)
        
        # Save model for API
        print("\nSaving model for API...")
        model_path = args.output_path
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save with metadata
        model_data = {
            'model': final_model,
            'preprocessor': preprocessor,
            'feature_names': feature_names,
            'model_version': '1.0.0',
            'model_type': args.model_type,
            'best_threshold': best_threshold,
            'best_f1': best_f1,
            'dataset_version': DATASET_VERSION,
            'feature_set': FEATURES,
            'mlflow_run_id': mlflow.active_run().info.run_id,
        }
        
        with open(model_path, 'wb') as f:
            joblib.dump(model_data, f)
        
        print(f"Model saved to {model_path}")
        
        # Register model in MLflow Model Registry
        if args.register:
            print("\nRegistering model in MLflow Model Registry...")
            from mlflow.tracking import MlflowClient
            
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/model"
            
            registered_name = f"{args.experiment_name}_{args.model_type}_api"
            
            mv = mlflow.register_model(model_uri, registered_name)
            
            client = MlflowClient()
            try:
                client.transition_model_version_stage(
                    name=registered_name,
                    version=mv.version,
                    stage="Staging"
                )
                print(f"Model registered: {registered_name} v{mv.version} (Staging)")
            except Exception as e:
                print(f"Warning: Could not transition to Staging: {e}")
            
            mlflow.set_tag("registered_model", registered_name)
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    train_and_save()
