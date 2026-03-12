import numpy as np
import pandas as pd
import pytest

from features.labeling import generate_labels
from config import LOOKAHEAD


def _make_df(closes: list) -> pd.DataFrame:
    index = pd.date_range('2026-01-01', periods=len(closes), freq='1min', tz='UTC')
    return pd.DataFrame({'close': closes}, index=index)


def test_label_up():
    """close[t+5] > close[t] → y[t] = 1"""
    # t=4: close=100, t+5=9: close=110  (indices 0-8 → 100, indices 9-19 → 110)
    closes = [100.0] * 9 + [110.0] * 11
    df = _make_df(closes)
    labels = generate_labels(df, lookahead=5)
    assert labels.iloc[4] == 1, f"Attendu 1, obtenu {labels.iloc[4]}"


def test_label_down():
    """close[t+5] < close[t] → y[t] = 0"""
    # t=4: close=100, t+5=9: close=90  (indices 0-8 → 100, indices 9-19 → 90)
    closes = [100.0] * 9 + [90.0] * 11
    df = _make_df(closes)
    labels = generate_labels(df, lookahead=5)
    assert labels.iloc[4] == 0, f"Attendu 0, obtenu {labels.iloc[4]}"


def test_last_rows_nan():
    """Les LOOKAHEAD dernières lignes doivent avoir NaN."""
    closes = [100.0] * 20
    df = _make_df(closes)
    labels = generate_labels(df, lookahead=LOOKAHEAD)
    assert labels.iloc[-LOOKAHEAD:].isna().all(), "Les dernières lignes doivent être NaN"


def test_non_nan_rows():
    """Les premières lignes (hors lookahead final) ne doivent pas être NaN."""
    closes = [float(i) for i in range(1, 21)]
    df = _make_df(closes)
    labels = generate_labels(df, lookahead=LOOKAHEAD)
    # Les n-LOOKAHEAD premières lignes doivent avoir des valeurs
    assert not labels.iloc[:-LOOKAHEAD].isna().any(), "Les lignes hors lookahead ne doivent pas être NaN"
