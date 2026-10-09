"""
Pipeline extracted from the notebook `notebooks/EDA_data_processing.ipynb`.

This module provides:
- load_application_files(): Load train/test CSVs
- encode_categoricals(): Binary label encoding + one-hot encoding for others
- add_polynomial_features(): Create polynomial features from EXT_SOURCE_* and DAYS_BIRTH
- add_domain_features(): Create domain ratio features
- train_model(): Generic model training with K-Fold CV (supports any scikit-learn estimator)
- tune_optuna(): Optimize hyperparameters using Optuna with manual K-Fold CV
- get_*_model_config(): Helper functions to initialize different models (LightGBM, LogisticRegression, RandomForest)
"""

import re
import numpy as np
import pandas as pd
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import optuna
from optuna_integration.lightgbm import LightGBMPruningCallback

import mlflow


from pathlib import Path


def ensure_tracking_dir(tracking_uri: str) -> None:
    """Ensure the parent directory of a relative sqlite URI exists.

    SQLite creates the database file on demand but not its parent folder,
    so 'sqlite:///tracking/mlflow.db' would fail if tracking/ is missing.
    Absolute sqlite URIs (four slashes) and non-sqlite URIs are left alone.
    """
    prefix = "sqlite:///"
    if tracking_uri.startswith(prefix) and not tracking_uri.startswith("sqlite:////"):
        Path(tracking_uri[len(prefix):]).parent.mkdir(parents=True, exist_ok=True)


def setup_mlflow_experiment(experiment_name: str, tracking_uri: str) -> None:
    """Configure MLflow and make sure the experiment exists.

    Wraps ensure_tracking_dir + set_tracking_uri + set_experiment. When the
    experiment must be created (fresh mlflow.db), its artifact location is
    set explicitly inside tracking/mlruns/ — the SQLite default would
    otherwise scatter new artifacts back into ./mlruns at the project root.
    Remote (non-sqlite) tracking servers keep their own default location.
    """
    ensure_tracking_dir(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    if client.get_experiment_by_name(experiment_name) is None:
        if tracking_uri.startswith("sqlite:///"):
            artifact_location = Path("tracking/mlruns", experiment_name).resolve().as_uri()
            client.create_experiment(experiment_name, artifact_location=artifact_location)
        else:
            client.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)


def load_application_files(data_dir="../data/raw", nrows=None):
    """Load `application_train.csv` and `application_test.csv` into dataframes.

    Args:
        data_dir (str): path to folder containing CSVs.
        nrows (int or None): if provided, read only this many rows (useful for debugging).

    Returns:
        tuple(pd.DataFrame, pd.DataFrame): (app_train, app_test)
    """
    train_path = f"{data_dir}/application_train.csv"
    test_path = f"{data_dir}/application_test.csv"
    app_train = pd.read_csv(train_path, nrows=nrows)
    app_test = pd.read_csv(test_path, nrows=nrows)
    return app_train, app_test


def encode_categoricals(app_train, app_test):
    """Encode categorical variables following the notebook policy:
    - If a categorical column has 2 or fewer uniques -> label encode
    - Otherwise -> one-hot encode (pd.get_dummies)

    This function returns modified copies of the inputs.
    """
    # Work on copies to avoid mutating caller data
    train = app_train.copy()
    test = app_test.copy()

    # Preserve identifier and target columns before one-hot encoding
    train_ids = train['SK_ID_CURR']
    train_target = train['TARGET']
    test_ids = test['SK_ID_CURR']
    train = train.drop(columns=['SK_ID_CURR', 'TARGET'])
    test = test.drop(columns=['SK_ID_CURR'])

    # Label encode binary categorical columns using the combined train/test values so
    # test can contain labels not seen in the train subset used for debugging.
    for col in train.columns:
        if train[col].dtype == 'object':
            combined_values = pd.concat([train[col], test[col]], axis=0).astype(str)
            unique_values = combined_values.unique()
            if len(unique_values) <= 2:
                le = LabelEncoder()
                le.fit(unique_values)
                train[col] = le.transform(train[col].astype(str))
                test[col] = le.transform(test[col].astype(str))

    # One-hot encode remaining categoricals using pandas.get_dummies
    train = pd.get_dummies(train).astype(float)
    test = pd.get_dummies(test).astype(float)

    # Align train and test feature columns and fill missing categories with 0
    train, test = train.align(test, join='outer', axis=1, fill_value=0)

    # Reattach ID and target columns
    train['SK_ID_CURR'] = train_ids.values
    train['TARGET'] = train_target.values
    test['SK_ID_CURR'] = test_ids.values

    return train, test


def add_polynomial_features(app_train, app_test, degree=3):
    """Create polynomial features from the EXT_SOURCEs and DAYS_BIRTH as in the notebook.

    Returns copies of app_train and app_test with new polynomial features merged in.
    """
    # Select the columns used for polynomial features in the notebook
    cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']

    # Ensure columns exist; if missing, skip this step
    for c in cols:
        if c not in app_train.columns:
            return app_train.copy(), app_test.copy()

    # Subset data for polynomial creation
    poly_train = app_train[cols + ['SK_ID_CURR', 'TARGET']].copy()
    poly_test = app_test[cols + ['SK_ID_CURR']].copy()

    # Impute missing values with median (SimpleImputer)
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(poly_train[cols])
    X_test = imputer.transform(poly_test[cols])

    # Polynomial transform
    poly = PolynomialFeatures(degree=degree)
    poly.fit(X_train)
    Xp_train = poly.transform(X_train)
    Xp_test = poly.transform(X_test)

    # Convert to DataFrames with feature names
    feature_names = poly.get_feature_names_out(cols)
    Xp_train_df = pd.DataFrame(Xp_train, columns=feature_names, index=poly_train.index)
    Xp_test_df = pd.DataFrame(Xp_test, columns=feature_names, index=poly_test.index)

    # Attach SK_ID_CURR and merge back to original
    Xp_train_df['SK_ID_CURR'] = poly_train['SK_ID_CURR'].values
    Xp_test_df['SK_ID_CURR'] = poly_test['SK_ID_CURR'].values

    app_train_poly = app_train.merge(Xp_train_df, on='SK_ID_CURR', how='left')
    app_test_poly = app_test.merge(Xp_test_df, on='SK_ID_CURR', how='left')

    # Preserve IDs and target before alignment
    train_ids = app_train_poly['SK_ID_CURR']
    train_target = app_train_poly['TARGET']
    test_ids = app_test_poly['SK_ID_CURR']

    train_features = app_train_poly.drop(columns=['SK_ID_CURR', 'TARGET'])
    test_features = app_test_poly.drop(columns=['SK_ID_CURR'])

    # Align only the feature columns between train and test
    train_features, test_features = train_features.align(test_features, join='inner', axis=1)

    train_features['SK_ID_CURR'] = train_ids.values
    train_features['TARGET'] = train_target.values
    test_features['SK_ID_CURR'] = test_ids.values

    return train_features, test_features


def _add_domain_features(df):
    """Add the domain ratio features to a single dataframe (train or test).

    This is the shared implementation used at training time (via
    add_domain_features) and at serving time (via CreditPreprocessor.transform).
    """
    out = df.copy()

    if 'AMT_CREDIT' in out.columns and 'AMT_INCOME_TOTAL' in out.columns:
        out['CREDIT_INCOME_PERCENT'] = out['AMT_CREDIT'] / out['AMT_INCOME_TOTAL']
    if 'AMT_ANNUITY' in out.columns and 'AMT_INCOME_TOTAL' in out.columns:
        out['ANNUITY_INCOME_PERCENT'] = out['AMT_ANNUITY'] / out['AMT_INCOME_TOTAL']
    if 'AMT_ANNUITY' in out.columns and 'AMT_CREDIT' in out.columns:
        out['CREDIT_TERM'] = out['AMT_ANNUITY'] / out['AMT_CREDIT']
    if 'DAYS_EMPLOYED' in out.columns and 'DAYS_BIRTH' in out.columns:
        out['DAYS_EMPLOYED_PERCENT'] = out['DAYS_EMPLOYED'] / out['DAYS_BIRTH']

    return out


def add_domain_features(app_train, app_test):
    """Add simple domain features used in the notebook (ratios and percentages).

    These are:
    - CREDIT_INCOME_PERCENT = AMT_CREDIT / AMT_INCOME_TOTAL
    - ANNUITY_INCOME_PERCENT = AMT_ANNUITY / AMT_INCOME_TOTAL
    - CREDIT_TERM = AMT_ANNUITY / AMT_CREDIT
    - DAYS_EMPLOYED_PERCENT = DAYS_EMPLOYED / DAYS_BIRTH
    """
    return _add_domain_features(app_train), _add_domain_features(app_test)


# ============================================================================
# MODEL INITIALIZATION HELPERS
# ============================================================================

def get_lightgbm_model(**kwargs):
    """Create a LightGBM classifier with given hyperparameters.

    Default params (if not overridden in kwargs):
        n_estimators=10000, objective='binary', class_weight='balanced',
        learning_rate=0.05, reg_alpha=0.1, reg_lambda=0.1, subsample=0.8,
        n_jobs=-1, random_state=50

    Args:
        **kwargs: override any default parameter

    Returns:
        lgb.LGBMClassifier instance
    """
    defaults = {
        'n_estimators': 10000,
        'objective': 'binary',
        'class_weight': 'balanced',
        'learning_rate': 0.05,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'subsample': 0.8,
        'n_jobs': -1,
        'random_state': 50
    }
    defaults.update(kwargs)
    return lgb.LGBMClassifier(**defaults)


def get_logistic_regression_model(**kwargs):
    """Create a LogisticRegression classifier.

    Default params (if not overridden in kwargs):
        C=0.1, max_iter=1000, random_state=50

    Args:
        **kwargs: override any default parameter

    Returns:
        LogisticRegression instance
    """
    defaults = {
        'C': 0.1,
        'max_iter': 1000,
        'random_state': 50
    }
    defaults.update(kwargs)
    return LogisticRegression(**defaults)


def get_random_forest_model(**kwargs):
    """Create a RandomForestClassifier.

    Default params (if not overridden in kwargs):
        n_estimators=100, n_jobs=-1, random_state=50, verbose=0

    Args:
        **kwargs: override any default parameter

    Returns:
        RandomForestClassifier instance
    """
    defaults = {
        'n_estimators': 100,
        'n_jobs': -1,
        'random_state': 50,
        'verbose': 0
    }
    defaults.update(kwargs)
    return RandomForestClassifier(**defaults)


def _sanitize_feature_names(columns):
    return [re.sub(r"[^0-9A-Za-z_]+", "_", str(col)).strip("_") for col in columns]


def _build_model(model_type, **kwargs):
    if model_type == 'lightgbm':
        return get_lightgbm_model(**kwargs)
    elif model_type == 'logistic_regression':
        return get_logistic_regression_model(**kwargs)
    elif model_type == 'random_forest':
        return get_random_forest_model(**kwargs)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. Must be 'lightgbm', 'logistic_regression', or 'random_forest'"
        )


def _prepare_training_data(features, test_features):
    features = features.copy()
    test_features = test_features.copy()

    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']
    labels = features['TARGET'].values

    features = features.drop(columns=['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns=['SK_ID_CURR'])

    features = features.apply(pd.to_numeric, errors='coerce').astype(float)
    test_features = test_features.apply(pd.to_numeric, errors='coerce').astype(float)

    feature_names = _sanitize_feature_names(features.columns)
    features.columns = feature_names
    test_features.columns = feature_names

    return train_ids, test_ids, labels, np.array(features), np.array(test_features), feature_names


# ============================================================================
# MODEL TRAINING
# ============================================================================
def compute_cost(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    return 10 * fn + fp

def apply_threshold(submission, threshold):
    submission = submission.copy()
    submission["TARGET"] = (submission["TARGET"] >= threshold).astype(int)
    return submission

def train_model(features, test_features, model_type='lightgbm', model_kwargs=None, n_folds=3):
    """Train a model using K-Fold CV and return submission / feature importances / metrics.

    This function works with any scikit-learn compatible estimator. It handles:
    - Stratified K-Fold cross-validation
    - Feature importance extraction (if available)
    - Out-of-fold predictions for validation
    - Support for multiple model types: LightGBM, LogisticRegression, RandomForest

    Args:
        features (pd.DataFrame): Training dataframe with SK_ID_CURR, TARGET, and features
        test_features (pd.DataFrame): Test dataframe with SK_ID_CURR and features
        model_type (str): Type of model: 'lightgbm', 'logistic_regression', or 'random_forest'
        model_kwargs (dict): Hyperparameters to pass to the model. If None, defaults are used.
        n_folds (int): Number of CV folds

    Returns:
        tuple: (submission, feature_importances, metrics, final_model)
            - submission: DataFrame with SK_ID_CURR and TARGET predictions
            - feature_importances: DataFrame with feature names and importances (or None if not available)
            - metrics: DataFrame with train/valid AUC per fold and overall
            - final_model: model retrained on the full training set
    """
    if model_kwargs is None:
        model_kwargs = {}

    train_ids, test_ids, labels, X, X_test, feature_names = _prepare_training_data(features, test_features)

    k_fold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=50)

    feature_importance_values = np.zeros(len(feature_names))
    test_predictions = np.zeros(X_test.shape[0])
    out_of_fold = np.zeros(X.shape[0])

    valid_scores = []
    train_scores = []

    use_imputer = model_type in {'logistic_regression', 'random_forest'}

    for train_indices, valid_indices in k_fold.split(X, labels):
        train_X, train_y = X[train_indices], labels[train_indices]
        valid_X, valid_y = X[valid_indices], labels[valid_indices]

        if use_imputer:
            imputer = SimpleImputer(strategy='median')
            train_X = imputer.fit_transform(train_X)
            valid_X = imputer.transform(valid_X)
            X_test_fold = imputer.transform(X_test)
        else:
            X_test_fold = X_test

        model = _build_model(model_type, **model_kwargs)

        if model_type == 'lightgbm':
            train_X_df = pd.DataFrame(train_X, columns=feature_names)
            valid_X_df = pd.DataFrame(valid_X, columns=feature_names)
            X_test_fold_df = pd.DataFrame(X_test_fold, columns=feature_names)

            model.fit(
                train_X_df, train_y,
                eval_metric='auc',
                eval_set=[(valid_X_df, valid_y), (train_X_df, train_y)],
                eval_names=['valid', 'train'],
                callbacks=[
                    lgb.early_stopping(100),
                    lgb.log_evaluation(200)
                ]
            )
            best_iteration = model.best_iteration_
            out_of_fold[valid_indices] = model.predict_proba(valid_X_df, num_iteration=best_iteration)[:, 1]
            test_predictions += model.predict_proba(X_test_fold_df, num_iteration=best_iteration)[:, 1] / n_folds
            feature_importance_values += model.feature_importances_ / n_folds
            valid_score = model.best_score_['valid']['auc']
            train_score = model.best_score_['train']['auc']
        else:
            model.fit(train_X, train_y)
            valid_pred = model.predict_proba(valid_X)[:, 1]
            train_pred = model.predict_proba(train_X)[:, 1]
            out_of_fold[valid_indices] = valid_pred
            test_predictions += model.predict_proba(X_test_fold)[:, 1] / n_folds

            if hasattr(model, 'feature_importances_'):
                feature_importance_values += model.feature_importances_ / n_folds

            valid_score = roc_auc_score(valid_y, valid_pred)
            train_score = roc_auc_score(train_y, train_pred)

        valid_scores.append(valid_score)
        train_scores.append(train_score)

    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})

    if np.sum(feature_importance_values) > 0:
        feature_importances = pd.DataFrame(
            {'feature': feature_names, 'importance': feature_importance_values}
        ).sort_values('importance', ascending=False)
    else:
        feature_importances = None

    # =========================
    # V1 : F1 + BEST THRESHOLD
    # =========================
    precisions, recalls, thresholds = precision_recall_curve(labels, out_of_fold)

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

    best_idx = np.nanargmax(f1_scores)

    best_precision = float(round(precisions[best_idx], 4))
    best_recall = float(round(recalls[best_idx], 4))
    best_f1 = float(round(f1_scores[best_idx], 4))

    # threshold aligné correctement
    best_threshold = 0.5 if best_idx == 0 else float(round(thresholds[best_idx - 1], 4))

    # =========================
    # V2 : COST-BASED THRESHOLD OPTIMIZATION
    # =========================

    # thresholds = np.linspace(0.01, 0.99, 200)

    # costs = []

    # for t in thresholds:
    #     y_pred = (out_of_fold >= t).astype(int)
    #     cost = compute_cost(labels, y_pred)
    #     costs.append(cost)

    # best_idx = int(np.argmin(costs))
    # best_threshold = float(thresholds[best_idx])

    # # predictions finalisées avec seuil optimal
    # final_oof_pred = (out_of_fold >= best_threshold).astype(int)

    # # recompute metrics at best threshold
    # tn = np.sum((labels == 0) & (final_oof_pred == 0))
    # fp = np.sum((labels == 0) & (final_oof_pred == 1))
    # fn = np.sum((labels == 1) & (final_oof_pred == 0))
    # tp = np.sum((labels == 1) & (final_oof_pred == 1))

    # best_cost = 10 * fn + fp

    # best_precision = tp / (tp + fp + 1e-9)
    # best_recall = tp / (tp + fn + 1e-9)
    # best_f1 = 2 * best_precision * best_recall / (best_precision + best_recall + 1e-9)

    # =========================
    # GLOBAL METRICS
    # =========================
    valid_auc = roc_auc_score(labels, out_of_fold)

    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))


    fold_names = list(range(n_folds)) + ['overall']
    metrics = pd.DataFrame({
        'fold': fold_names,
        'train': train_scores,
        'valid': valid_scores
    })
    metrics[['train', 'valid']] = metrics[['train', 'valid']].round(4)

    final_model = _build_model(model_type, **model_kwargs)
    if use_imputer:
        imputer = SimpleImputer(strategy='median')
        X_full = imputer.fit_transform(X)
    else:
        X_full = X

    if model_type == 'lightgbm':
        final_model.fit(pd.DataFrame(X_full, columns=feature_names), labels)
    else:
        final_model.fit(X_full, labels)

    return submission, feature_importances, metrics, final_model, best_recall, best_precision, float(best_threshold), best_f1 # , float(best_cost)

def tune_optuna(features, n_trials=10, n_folds=3, model_type='lightgbm', random_state=50):

    df = features.copy()

    if 'TARGET' not in df.columns:
        raise ValueError('features must contain TARGET column')

    y = df['TARGET'].values
    X_df = df.drop(columns=['SK_ID_CURR', 'TARGET'])

    X_df = X_df.apply(pd.to_numeric, errors='coerce').astype(float)
    X_df.columns = [re.sub(r"[^0-9A-Za-z_]+", "_", str(col)).strip("_") for col in X_df.columns]
    X = X_df.values

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # =========================
    # OPTUNA STUDY + PRUNER
    # =========================
    ensure_tracking_dir("sqlite:///tracking/optuna.db")
    study = optuna.create_study(
        study_name=f'credit_scoring_optuna_{model_type}',
        storage="sqlite:///tracking/optuna.db",
        direction='maximize',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1)
    )

    def objective(trial):

        if model_type == 'lightgbm':
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 16, 128, step=8),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 50.0, log=True),
                'n_jobs': -1,
                'random_state': random_state
            }

        elif model_type == 'logistic_regression':
            params = {
                'C': trial.suggest_float('C', 1e-4, 100.0, log=True),
                'max_iter': 1000
            }

        elif model_type == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'n_jobs': -1
            }

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        oof = np.zeros(X.shape[0])
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_df, y)):

            X_tr, y_tr = X_df.iloc[train_idx], y[train_idx]
            X_val, y_val = X_df.iloc[val_idx], y[val_idx]

            if model_type in {'logistic_regression', 'random_forest'}:
                imputer = SimpleImputer(strategy='median')
                X_tr = imputer.fit_transform(X_tr)
                X_val = imputer.transform(X_val)

            # =========================
            # LIGHTGBM + PRUNING
            # =========================
            if model_type == 'lightgbm':

                clf = get_lightgbm_model(**params)

                clf.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric='auc',
                    callbacks=[
                        lgb.early_stopping(50),
                        lgb.log_evaluation(0),
                        optuna.integration.LightGBMPruningCallback(trial, "auc")
                    ]
                )

                preds = clf.predict_proba(X_val, num_iteration=clf.best_iteration_)[:, 1]

            else:
                if model_type == 'logistic_regression':
                    clf = get_logistic_regression_model(**params)
                else:
                    clf = get_random_forest_model(**params)

                clf.fit(X_tr, y_tr)
                preds = clf.predict_proba(X_val)[:, 1]

            oof[val_idx] = preds

            fold_score = roc_auc_score(y_val, preds)
            fold_scores.append(fold_score)

            # =========================
            # PRUNING STEP
            # =========================
            trial.report(np.mean(fold_scores), fold)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(roc_auc_score(y, oof))

    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params.copy()

    # cast int
    for key in ['num_leaves', 'max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf']:
        if key in best_params:
            best_params[key] = int(best_params[key])

    return study, best_params

# def tune_optuna(features, n_trials=10, n_folds=3, model_type='lightgbm', random_state=50):
#     """Optimize model hyperparameters using Optuna with manual K-Fold CV.

#     This function runs a K-Fold CV inside each Optuna trial, which allows
#     early stopping (for LightGBM) to be applied per fold and preserves OOF semantics.

#     Args:
#         features (pd.DataFrame): Dataframe with SK_ID_CURR, TARGET, and features (already encoded)
#         n_trials (int): Number of Optuna trials
#         n_folds (int): Number of CV folds
#         model_type (str): 'lightgbm', 'logistic_regression', or 'random_forest'
#         random_state (int): Random seed

#     Returns:
#         tuple: (study, best_params)
#             - study: Completed optuna.Study object
#             - best_params: Dictionary with best hyperparameters found
#     """
#     df = features.copy()

#     if 'TARGET' not in df.columns:
#         raise ValueError('features must contain TARGET column')

#     y = df['TARGET'].values
#     X_df = df.drop(columns=['SK_ID_CURR', 'TARGET'])
#     X_df = X_df.apply(pd.to_numeric, errors='coerce').astype(float)
#     X_df.columns = [re.sub(r"[^0-9A-Za-z_]+", "_", str(col)).strip("_") for col in X_df.columns]
#     X = X_df.values

#     kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

#     def objective(trial):
#         """Objective function for Optuna trials."""
#         if model_type == 'lightgbm':
#             # Suggest LightGBM hyperparameters
#             params = {
#                 'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
#                 'num_leaves': trial.suggest_int('num_leaves', 16, 128, step=8),
#                 'max_depth': trial.suggest_int('max_depth', 3, 12),
#                 'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
#                 'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
#                 'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#                 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#                 'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 50.0, log=True)
#             }
#         elif model_type == 'logistic_regression':
#             # Suggest LogisticRegression hyperparameters
#             params = {
#                 'C': trial.suggest_float('C', 1e-4, 100.0, log=True),
#                 'max_iter': 1000
#             }
#         elif model_type == 'random_forest':
#             # Suggest RandomForest hyperparameters
#             params = {
#                 'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
#                 'max_depth': trial.suggest_int('max_depth', 5, 30),
#                 'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
#                 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
#             }
#         else:
#             raise ValueError(f"Unknown model_type: {model_type}")

#         # Run K-Fold CV and collect OOF predictions
#         oof = np.zeros(X.shape[0])

#         for train_idx, val_idx in kf.split(X_df, y):
#             X_tr, y_tr = X_df.iloc[train_idx], y[train_idx]
#             X_val, y_val = X_df.iloc[val_idx], y[val_idx]

#             # Impute missing values for sklearn estimators that do not accept NaN
#             if model_type in {'logistic_regression', 'random_forest'}:
#                 imputer = SimpleImputer(strategy='median')
#                 X_tr = imputer.fit_transform(X_tr)
#                 X_val = imputer.transform(X_val)

#             # Create and train model
#             if model_type == 'lightgbm':
#                 clf = get_lightgbm_model(**params)
#                 clf.fit(X_tr, y_tr,
#                         eval_set=[(X_val, y_val)],
#                         eval_metric='auc',
#                         callbacks=[
#                             lgb.early_stopping(50),
#                             lgb.log_evaluation(200)
#                         ])
#                 oof[val_idx] = clf.predict_proba(X_val, num_iteration=clf.best_iteration_)[:, 1]
#             elif model_type == 'logistic_regression':
#                 clf = get_logistic_regression_model(**params)
#                 clf.fit(X_tr, y_tr)
#                 oof[val_idx] = clf.predict_proba(X_val)[:, 1]
#             elif model_type == 'random_forest':
#                 clf = get_random_forest_model(**params)
#                 clf.fit(X_tr, y_tr)
#                 oof[val_idx] = clf.predict_proba(X_val)[:, 1]

#         # Compute OOF AUC
#         score = roc_auc_score(y, oof)
#         return float(score)

#     # Run Optuna study
#     study = optuna.create_study(study_name='credit_scoring_optuna', storage="sqlite:///tracking/optuna.db", direction='maximize', load_if_exists=True)
#     study.optimize(objective, n_trials=n_trials)

#     # Extract and format best params
#     best_params = study.best_trial.params.copy()

#     # Convert integer suggestions to int (not float)
#     for key in ['num_leaves', 'max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf']:
#         if key in best_params:
#             best_params[key] = int(best_params[key])

#     return study, best_params


# ============================================================================
# INFERENCE PREPROCESSOR (fitted at training time, reused at serving time)
# ============================================================================

class CreditPreprocessor:
    """Preprocessing pipeline mirroring the exact training transformations.

    Replicates the notebook chain on any raw application row:

        raw columns -> polynomial features (EXT_SOURCE_*, DAYS_BIRTH)
                    -> domain ratio features
                    -> label / one-hot encoding
                    -> numeric coercion + sanitized feature order

    The fitted state (polynomial medians, label mappings, one-hot columns,
    final feature order) is plain Python data, so the instance serializes
    cleanly with joblib inside the model artifact. At serving time,
    transform() produces the exact feature frame the model was trained on:

    - label-encoded categories: unseen values map to -1
    - one-hot categories: unseen values produce all-zero dummies
    - missing raw columns: filled with 0 after feature ordering
    """

    POLY_COLS = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']

    def __init__(self, degree=3):
        self.degree = degree
        self.poly_medians = None        # {col: median} fitted on train
        self.poly_feature_names = None  # PolynomialFeatures output names
        self.label_maps = {}            # {col: {value: code}}
        self.onehot_columns = []        # object columns one-hot encoded
        self.feature_names_ = None      # final sanitized feature order

    def fit(self, app_train, app_test=None):
        """Fit the preprocessing state on application data.

        Args:
            app_train: application_train DataFrame (requires SK_ID_CURR and TARGET)
            app_test: optional application_test DataFrame. When provided, the
                label-encoding rule uses combined train+test values, matching
                the behavior of encode_categoricals at training time.
        """
        train = app_train.copy()
        if 'SK_ID_CURR' not in train.columns or 'TARGET' not in train.columns:
            raise ValueError('app_train must contain SK_ID_CURR and TARGET')

        # Polynomial state: medians fitted on train, same as add_polynomial_features
        if all(c in train.columns for c in self.POLY_COLS):
            imputer = SimpleImputer(strategy='median')
            X_poly = imputer.fit_transform(train[self.POLY_COLS])
            self.poly_medians = dict(zip(self.POLY_COLS, imputer.statistics_))
            poly = PolynomialFeatures(degree=self.degree)
            poly.fit(X_poly)
            self.poly_feature_names = list(poly.get_feature_names_out(self.POLY_COLS))
        else:
            self.poly_medians = None
            self.poly_feature_names = None

        # Encoding state: same rule as encode_categoricals
        # (object columns with <= 2 uniques are label encoded, others one-hot)
        other = train if app_test is None else app_test
        for col in train.columns:
            if train[col].dtype == 'object':
                combined = pd.concat([train[col], other[col]], axis=0).astype(str)
                unique_values = sorted(combined.unique())
                if len(unique_values) <= 2:
                    self.label_maps[col] = {value: code for code, value in enumerate(unique_values)}
                else:
                    self.onehot_columns.append(col)

        # Final feature order: replicate the full training chain,
        # propagating the transformed test frame at each step (like
        # train_model.py does), so align(outer) sees identical columns
        tr_poly, te_poly = add_polynomial_features(train, other)
        tr_fe, te_fe = add_domain_features(tr_poly, te_poly)
        tr_enc, _ = encode_categoricals(tr_fe, te_fe)
        ordered = [c for c in tr_enc.columns if c not in ('SK_ID_CURR', 'TARGET')]
        self.feature_names_ = _sanitize_feature_names(ordered)

        return self

    def transform(self, df):
        """Transform raw application row(s) into the model feature frame.

        Args:
            df: DataFrame of raw application columns (partial rows allowed)

        Returns:
            pd.DataFrame with columns == feature_names_, ready for model.predict
        """
        if self.feature_names_ is None:
            raise RuntimeError('CreditPreprocessor must be fitted before transform')

        X = df.copy()

        # 1. Label-encoded categories (unseen values -> -1)
        for col, mapping in self.label_maps.items():
            if col in X.columns:
                X[col] = X[col].astype(str).map(mapping).fillna(-1)
            else:
                X[col] = -1

        # 2. One-hot categories (dummies re-aligned later via feature order)
        if self.onehot_columns:
            present = [c for c in self.onehot_columns if c in X.columns]
            if present:
                dummies = pd.get_dummies(X[present].astype(str))
                X = pd.concat([X.drop(columns=present), dummies], axis=1)

        # 3. Numeric coercion (same as _prepare_training_data)
        X = X.apply(pd.to_numeric, errors='coerce').astype(float)

        # 4. Polynomial features (imputed with the medians fitted on train)
        if self.poly_feature_names:
            sub = X.reindex(columns=self.POLY_COLS, fill_value=np.nan)
            sub = sub.fillna(pd.Series(self.poly_medians))
            poly = PolynomialFeatures(degree=self.degree)
            Xp = poly.fit_transform(sub)
            Xp_df = pd.DataFrame(Xp, columns=self.poly_feature_names, index=X.index)
            # Replicate the merge suffixing of add_polynomial_features:
            # colliding columns become <col>_x (original) and <col>_y (poly).
            # Side effect kept for parity: raw DAYS_BIRTH becomes DAYS_BIRTH_x,
            # so the DAYS_EMPLOYED_PERCENT domain feature is skipped at serving
            # time exactly as it is at training time.
            collisions = [c for c in self.poly_feature_names if c in X.columns]
            if collisions:
                X = X.rename(columns={c: f"{c}_x" for c in collisions})
                Xp_df = Xp_df.rename(columns={c: f"{c}_y" for c in collisions})
            X = pd.concat([X, Xp_df], axis=1)

        # 5. Domain ratio features
        X = _add_domain_features(X)

        # 6. Sanitize names and align on the training feature order
        X.columns = _sanitize_feature_names(X.columns)
        X = X.reindex(columns=self.feature_names_, fill_value=0.0)

        return X.fillna(0.0)
