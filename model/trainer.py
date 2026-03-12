import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss
from lightgbm import LGBMClassifier, early_stopping

from config import N_SPLITS, GAP, LGBM_PARAMS, MODEL_PATH, SCALER_PATH
from model.calibrator import PlattCalibratedClassifier


def train_pipeline(X: np.ndarray, y: np.ndarray) -> None:
    """
    Entraîne le pipeline LightGBM + Platt Scaling.

    1. Walk-forward validation pour évaluer le modèle (Brier Score par fold)
    2. Entraînement final sur 60% fit + 20% calibration
    3. Sauvegarde du modèle calibré et du scaler
    """
    # --- Walk-forward validation ---
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=GAP)
    fold_scores = []

    print("\n=== Walk-Forward Validation ===")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_fold, y_train_fold = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Séparer fit (80%) et calibration (20%) du fold train
        # Minimum 50 pour que Platt Scaling soit fiable
        cal_size = max(50, int(0.2 * len(X_train_fold)))
        X_fit_f, X_cal_f = X_train_fold[:-cal_size], X_train_fold[-cal_size:]
        y_fit_f, y_cal_f = y_train_fold[:-cal_size], y_train_fold[-cal_size:]

        # Scaler fitté sur X_fit uniquement
        scaler_fold = StandardScaler()
        X_fit_s  = scaler_fold.fit_transform(X_fit_f)
        X_cal_s  = scaler_fold.transform(X_cal_f)
        X_val_s  = scaler_fold.transform(X_val)

        # LightGBM avec early stopping — eval_metric explicite (sklearn API)
        lgbm = LGBMClassifier(**LGBM_PARAMS, n_estimators=1000)
        lgbm.fit(
            X_fit_s, y_fit_f,
            eval_set=[(X_cal_s, y_cal_f)],
            eval_metric='binary_logloss',
            callbacks=[early_stopping(50, verbose=False)],
        )

        # Platt Scaling
        cal_model = PlattCalibratedClassifier(lgbm)
        cal_model.fit(X_cal_s, y_cal_f)

        probs = cal_model.predict_proba(X_val_s)[:, 1]
        bs = brier_score_loss(y_val, probs)
        fold_scores.append(bs)
        print(f"  Fold {fold + 1}: Brier Score = {bs:.4f}")

    print(f"  Mean Brier Score (walk-forward): {np.mean(fold_scores):.4f}\n")

    # --- Entraînement final ---
    print("=== Entraînement final ===")
    n = len(X)
    X_fit, y_fit = X[:int(0.6 * n)], y[:int(0.6 * n)]
    X_cal, y_cal = X[int(0.6 * n):int(0.8 * n)], y[int(0.6 * n):int(0.8 * n)]

    scaler = StandardScaler()
    X_fit_s = scaler.fit_transform(X_fit)
    X_cal_s = scaler.transform(X_cal)

    lgbm_final = LGBMClassifier(**LGBM_PARAMS, n_estimators=1000)
    lgbm_final.fit(
        X_fit_s, y_fit,
        eval_set=[(X_cal_s, y_cal)],
        eval_metric='binary_logloss',
        callbacks=[early_stopping(50, verbose=False)],
    )

    final_model = PlattCalibratedClassifier(lgbm_final)
    final_model.fit(X_cal_s, y_cal)

    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Modèle sauvegardé : {MODEL_PATH}")
    print(f"Scaler sauvegardé : {SCALER_PATH}")
