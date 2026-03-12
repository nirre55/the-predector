"""
Tests pour model/predictor.py.

Deux modes :
- Si models/ contient un vrai modèle entraîné → tests sur le vrai modèle
- Sinon → fixture qui crée un modèle minimal synthétique pour CI/CD
"""
import os
import numpy as np
import pytest
import joblib
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from model.calibrator import PlattCalibratedClassifier

from config import MODEL_PATH, SCALER_PATH

N_FEATURES = 25  # nombre de features produit par compute_features


def _build_minimal_model(model_path: str, scaler_path: str) -> None:
    """Crée et sauvegarde un modèle LightGBM minimal pour les tests."""
    rng = np.random.default_rng(0)
    X = rng.random((300, N_FEATURES))
    y = (rng.random(300) > 0.5).astype(int)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    lgbm = LGBMClassifier(n_estimators=10, verbosity=-1, n_jobs=1)
    lgbm.fit(X_s[:240], y[:240])

    cal = PlattCalibratedClassifier(lgbm)
    cal.fit(X_s[240:], y[240:])

    joblib.dump(cal, model_path)
    joblib.dump(scaler, scaler_path)


@pytest.fixture(scope='module')
def predictor(tmp_path_factory):
    """
    Retourne un Predictor chargé soit depuis le vrai modèle (si existant),
    soit depuis un modèle minimal créé pour les tests.
    """
    from model.predictor import Predictor

    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        return Predictor()

    # Créer un modèle minimal dans un répertoire temporaire
    tmp = tmp_path_factory.mktemp('models')
    tmp_model  = str(tmp / 'model.joblib')
    tmp_scaler = str(tmp / 'scaler.joblib')
    _build_minimal_model(tmp_model, tmp_scaler)

    return Predictor(model_path=tmp_model, scaler_path=tmp_scaler)


def test_predict_returns_float(predictor):
    """La sortie de predict() doit être un float."""
    n_features = predictor.scaler.n_features_in_
    feature_vec = np.random.rand(n_features)
    result = predictor.predict(feature_vec)
    assert isinstance(result, float), f"Attendu float, obtenu {type(result)}"


def test_predict_in_range(predictor):
    """La sortie doit être dans [0.001, 0.999]."""
    n_features = predictor.scaler.n_features_in_
    for _ in range(10):
        feature_vec = np.random.rand(n_features)
        result = predictor.predict(feature_vec)
        assert 0.001 <= result <= 0.999, f"Valeur hors plage : {result}"


def test_predict_accepts_1d_vector(predictor):
    """predict() doit accepter un vecteur 1D."""
    n_features = predictor.scaler.n_features_in_
    feature_vec = np.zeros(n_features)
    result = predictor.predict(feature_vec)
    assert isinstance(result, float)
