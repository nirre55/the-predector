from datetime import datetime

import pandas as pd
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


class LiveDisplay:
    """Affichage Rich temps réel mis à jour à chaque prédiction."""

    def __init__(self):
        self._console = Console()
        self._live    = Live(
            self._build_table(None, None, None, None, 0),
            console=self._console,
            refresh_per_second=4,
        )
        self._prediction_count = 0
        self._current_market_window: int | None = None  # minute // 5

    @property
    def live(self) -> Live:
        return self._live

    def update(
        self,
        prob_up: float,
        signal: str,
        edge: float,
        close_price: float,
        timestamp,
    ) -> None:
        """Met à jour l'affichage avec les dernières données."""
        ts = pd.Timestamp(timestamp)
        market_window = (ts.hour * 60 + ts.minute) // 5
        if market_window != self._current_market_window:
            self._current_market_window = market_window
            self._prediction_count = 0
        self._prediction_count += 1
        self._live.update(
            self._build_table(prob_up, signal, edge, close_price, timestamp)
        )

    def _build_table(
        self,
        prob_up,
        signal,
        edge,
        close_price,
        timestamp,
    ) -> Panel:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Champ",  style="bold", width=22)
        table.add_column("Valeur", width=30)

        if timestamp is not None:
            ts_str = pd.Timestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S UTC')
        else:
            ts_str = "En attente..."

        close_str = f"{close_price:,.2f} USDT" if close_price is not None else "—"
        prob_str  = f"{prob_up:.3f}"           if prob_up   is not None else "—"
        edge_str  = f"+{edge:.1f}%"            if edge      is not None else "—"

        if signal == 'UP':
            signal_str = "[bold green]▲ UP[/bold green]"
        elif signal == 'DOWN':
            signal_str = "[bold red]▼ DOWN[/bold red]"
        else:
            signal_str = "—"

        table.add_row("Heure",       ts_str)
        table.add_row("Close",       close_str)
        table.add_row("P(Up)",       prob_str)
        table.add_row("Signal",      signal_str)
        table.add_row("Edge",        edge_str)
        table.add_row("Prédictions", f"{self._prediction_count} ce marché")

        return Panel(table, title="[bold cyan]BTCUSDT — Polymarket Predictor[/bold cyan]", border_style="cyan")
