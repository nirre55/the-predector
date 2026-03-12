import numpy as np
import joblib


class Predictor:
    """Charge le modèle calibré et le scaler pour la prédiction live."""

    def __init__(self, model_path: str | None = None, scaler_path: str | None = None):
        from config import MODEL_PATH, SCALER_PATH
        self.model  = joblib.load(model_path  or MODEL_PATH)
        self.scaler = joblib.load(scaler_path or SCALER_PATH)

    def predict(self, feature_vector: np.ndarray) -> float:
        """
        Prédit P(up) pour un vecteur de features 1D.

        Args:
            feature_vector: np.ndarray 1D de shape (n_features,)

        Returns:
            float dans [0.001, 0.999] représentant P(close[t+5] > close[t])
        """
        X = self.scaler.transform(feature_vector.reshape(1, -1))
        prob_up = self.model.predict_proba(X)[0][1]
        return float(np.clip(prob_up, 0.001, 0.999))
