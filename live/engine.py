import asyncio
import logging

import numpy as np
import pandas as pd

from data.stream import BinanceStream
from features.engineering import compute_features
from live.display import LiveDisplay
from model.predictor import Predictor

logger = logging.getLogger(__name__)


async def _run_live_async() -> None:
    """Boucle principale asynchrone : stream → features → prédiction → affichage."""
    predictor = Predictor()
    display   = LiveDisplay()

    async def on_candle_close(candles: list) -> None:
        df = pd.DataFrame(candles).set_index('open_time')
        features, _, _ = compute_features(df)

        if len(features) == 0:
            return

        feature_vec = features[-1]
        if np.isnan(feature_vec).any():
            logger.debug("Feature vector contient des NaN — pas assez de données.")
            return

        prob_up = predictor.predict(feature_vec)
        signal  = 'UP' if prob_up >= 0.5 else 'DOWN'
        edge    = abs(prob_up - 0.5) * 200  # edge en %

        display.update(
            prob_up=prob_up,
            signal=signal,
            edge=edge,
            close_price=df['close'].iloc[-1],
            timestamp=df.index[-1],
        )

    stream = BinanceStream(on_candle_close)
    with display.live:
        await stream.run()


def run_live() -> None:
    """Point d'entrée synchrone pour le mode live."""
    asyncio.run(_run_live_async())
