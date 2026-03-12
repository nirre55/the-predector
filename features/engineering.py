import numpy as np
import pandas as pd
import pandas_ta as ta

from config import FEATURE_WINDOW


def compute_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str], pd.Index]:
    """
    Calcule les features sur un DataFrame OHLCV.

    Mode batch (entraînement) : DataFrame complet → retourne (array_2D, feature_names, index)
      Les premières lignes avec NaN sont supprimées. Utiliser l'index retourné pour
      aligner avec les labels (évite tout décalage silencieux).
    Mode live (prédiction) : buffer de FEATURE_WINDOW bougies → retourne (array_2D, feature_names, index)
      Utiliser features[-1] pour la dernière ligne.

    Returns:
        tuple[np.ndarray, list[str], pd.Index] : tableau de features, noms, index des lignes valides
    """
    feat = pd.DataFrame(index=df.index)

    # --- Returns ---
    feat['close_ret_1m']  = df['close'].pct_change(1)
    feat['close_ret_5m']  = df['close'].pct_change(5)
    feat['close_ret_15m'] = df['close'].pct_change(15)
    feat['close_ret_30m'] = df['close'].pct_change(30)
    feat['hl_range']      = (df['high'] - df['low']) / df['close']

    # --- Body / Wicks ---
    feat['body_size']   = (df['close'] - df['open']).abs() / df['close']
    feat['upper_wick']  = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    feat['lower_wick']  = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']

    # --- Volume ---
    vol_ma20 = df['volume'].rolling(20).mean()
    feat['volume_ratio'] = df['volume'] / vol_ma20
    # VWAP rolling sur 20 bougies (évite le cumul sur années entières)
    roll_vol = df['volume'].rolling(20).sum()
    vwap = (df['close'] * df['volume']).rolling(20).sum() / roll_vol.replace(0, np.nan)
    feat['vwap_dev'] = (df['close'] - vwap) / df['close']

    # --- Momentum (pandas-ta) ---
    feat['rsi_14'] = ta.rsi(df['close'], length=14) / 50 - 1  # type: ignore[attr-defined]

    macd_obj = ta.macd(df['close'], fast=12, slow=26, signal=9)  # type: ignore[attr-defined]
    feat['macd_cross']      = macd_obj['MACD_12_26_9'] - macd_obj['MACDs_12_26_9']
    feat['macd_hist_slope'] = macd_obj['MACDh_12_26_9'].diff(1)

    stoch_obj = ta.stoch(df['high'], df['low'], df['close'], k=5, d=3)  # type: ignore[attr-defined]
    # Accès positionnel : colonne 0 = STOCHk (varie selon version pandas-ta)
    feat['stoch_norm'] = stoch_obj.iloc[:, 0] / 50 - 1

    # --- Volatilité (pandas-ta) ---
    feat['atr_norm'] = ta.atr(df['high'], df['low'], df['close'], length=14) / df['close']  # type: ignore[attr-defined]

    bb = ta.bbands(df['close'], length=20, std=2)  # type: ignore[attr-defined, arg-type]
    # pandas-ta column naming varies by version — select by position to be robust
    bb_lower  = bb.iloc[:, 0]  # BBL
    bb_middle = bb.iloc[:, 1]  # BBM
    bb_upper  = bb.iloc[:, 2]  # BBU
    bb_range = bb_upper - bb_lower
    feat['bb_position'] = (df['close'] - bb_lower) / bb_range.replace(0, np.nan)
    feat['bb_width']    = bb_range / bb_middle

    pct_change = df['close'].pct_change()
    rv_5m  = pct_change.rolling(5).std()  * np.sqrt(5)
    rv_15m = pct_change.rolling(15).std() * np.sqrt(15)
    feat['rv_5m']      = rv_5m
    feat['rv_15m']     = rv_15m
    feat['vol_ratio']  = rv_5m / rv_15m.replace(0, np.nan)

    # --- Temporel ---
    dt_index = pd.DatetimeIndex(df.index)
    hour   = dt_index.hour
    minute = dt_index.minute  # noqa: F841 — disponible pour extensions futures
    feat['hour_sin']  = np.sin(2 * np.pi * hour / 24)
    feat['hour_cos']  = np.cos(2 * np.pi * hour / 24)
    feat['is_asia']   = ((hour >= 0) & (hour < 8)).astype(float)
    feat['is_europe'] = ((hour >= 8) & (hour < 16)).astype(float)
    feat['is_us']     = ((hour >= 13) & (hour < 22)).astype(float)

    feature_names = list(feat.columns)

    # Supprimer les lignes avec NaN (dues aux rolling windows)
    feat_clean = feat.dropna()

    return feat_clean.values, feature_names, feat_clean.index
