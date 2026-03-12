import pandas as pd
from config import LOOKAHEAD


def generate_labels(df: pd.DataFrame, lookahead: int = LOOKAHEAD) -> pd.Series:
    """
    Génère les labels binaires pour la prédiction de direction.

    y[t] = 1 si close[t+lookahead] > close[t], 0 sinon.
    Les `lookahead` dernières lignes ont NaN (pas de future close disponible).

    Returns:
        pd.Series avec le même index que df, dtype int (NaN sur les dernières lignes).
    """
    future_close = df['close'].shift(-lookahead)
    # Créer les labels uniquement où future_close est disponible
    labels = (future_close > df['close']).astype('Int64')
    labels = labels.where(~future_close.isna(), pd.NA)  # type: ignore[arg-type]
    return labels
