from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent

SYMBOL = 'BTCUSDT'
INTERVAL = '1m'
FEATURE_WINDOW = 60       # bougies 1m en fenêtre glissante
LOOKAHEAD = 5             # 5 bougies 1m = prédire dans 5m
TRAIN_YEARS = 2
DATA_PATH   = str(_PROJECT_ROOT / 'data' / 'raw' / 'BTCUSDT_1m.csv')
MODEL_PATH  = str(_PROJECT_ROOT / 'models' / 'pipeline_calibrated.joblib')
SCALER_PATH = str(_PROJECT_ROOT / 'models' / 'scaler.joblib')
WS_URL = 'wss://stream.binance.com:9443/ws/btcusdt@kline_1m'
REST_BASE = 'https://api.binance.com/api/v3/klines'
N_SPLITS = 5
GAP = 300

LGBM_PARAMS = {
    'objective': 'binary',
    'metric': ['binary_logloss', 'auc'],
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 50,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbosity': -1,
    'n_jobs': -1,
}
