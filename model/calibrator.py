import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibratedClassifier:
    """
    Platt Scaling : enveloppe un classificateur pré-entraîné avec calibration sigmoïde.

    Équivalent à CalibratedClassifierCV(estimator, method='sigmoid', cv='prefit')
    qui a été retiré dans scikit-learn >= 1.8.
    """

    def __init__(self, estimator):
        self.estimator = estimator
        self._platt = LogisticRegression()

    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray) -> 'PlattCalibratedClassifier':
        """Calibre sur les scores bruts du classificateur."""
        raw = self.estimator.predict_proba(X_cal)[:, 1].reshape(-1, 1)
        self._platt.fit(raw, y_cal)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retourne les probabilités calibrées — shape (n, 2)."""
        raw = self.estimator.predict_proba(X)[:, 1].reshape(-1, 1)
        return self._platt.predict_proba(raw)
