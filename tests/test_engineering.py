import numpy as np
import pandas as pd
import pytest

from features.engineering import compute_features


def _make_synthetic_df(n: int = 70) -> pd.DataFrame:
    """Crée un DataFrame OHLCV synthétique réaliste."""
    rng = np.random.default_rng(42)
    prices = 80000 + np.cumsum(rng.normal(0, 100, n))
    index = pd.date_range('2026-01-01', periods=n, freq='1min', tz='UTC')
    df = pd.DataFrame({
        'open':   prices + rng.uniform(-50, 50, n),
        'high':   prices + rng.uniform(50, 150, n),
        'low':    prices - rng.uniform(50, 150, n),
        'close':  prices,
        'volume': rng.uniform(1, 100, n),
    }, index=index)
    # S'assurer que high >= max(open, close) et low <= min(open, close)
    df['high'] = df[['open', 'high', 'close']].max(axis=1) + 10
    df['low']  = df[['open', 'low',  'close']].min(axis=1) - 10
    return df


def test_compute_features_returns_ndarray():
    df = _make_synthetic_df(70)
    features, names, idx = compute_features(df)
    assert isinstance(features, np.ndarray), "features doit être un np.ndarray"
    assert features.ndim == 2, "features doit être 2D"


def test_compute_features_at_least_one_clean_row():
    df = _make_synthetic_df(70)
    features, names, idx = compute_features(df)
    assert features.shape[0] >= 1, "Il doit y avoir au moins 1 ligne sans NaN"


def test_compute_features_count():
    df = _make_synthetic_df(70)
    features, names, idx = compute_features(df)
    # ~25 features (5 returns + 3 body/wicks + 2 volume + 4 momentum + 6 vol + 5 temporel)
    assert 20 <= features.shape[1] <= 40, f"Nombre de features inattendu : {features.shape[1]}"
    assert len(names) == features.shape[1], "len(names) doit correspondre au nombre de colonnes"


def test_compute_features_last_row_no_nan():
    df = _make_synthetic_df(70)
    features, names, idx = compute_features(df)
    last_row = features[-1]
    assert not np.isnan(last_row).any(), "La dernière ligne ne doit pas contenir de NaN"
