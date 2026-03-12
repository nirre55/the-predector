import warnings
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

warnings.filterwarnings('ignore', message='X does not have valid feature names', category=UserWarning)
from sklearn.calibration import calibration_curve
from rich.table import Table
from rich.console import Console


def evaluate(model, scaler, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Évalue le modèle calibré sur le jeu de test.

    Métriques : Brier Score, Brier Skill Score, Log Loss, AUC-ROC, ECE.

    Returns:
        dict avec les métriques calculées.
    """
    X_scaled = scaler.transform(X_test)
    probs = model.predict_proba(X_scaled)[:, 1]
    probs_clipped = np.clip(probs, 0.001, 0.999)

    bs  = brier_score_loss(y_test, probs_clipped)
    bss = 1 - bs / 0.25  # baseline naïve P=0.5 → BS=0.25
    ll  = log_loss(y_test, probs_clipped)
    auc = roc_auc_score(y_test, probs_clipped)

    # Expected Calibration Error
    frac_pos, mean_pred = calibration_curve(
        y_test, probs_clipped, n_bins=10, strategy='quantile'
    )
    ece = float(np.mean(np.abs(frac_pos - mean_pred)))

    metrics = {
        'brier_score':       bs,
        'brier_skill_score': bss,
        'log_loss':          ll,
        'auc_roc':           auc,
        'ece':               ece,
    }

    _print_metrics(metrics)
    return metrics


def _print_metrics(metrics: dict) -> None:
    """Affiche les métriques dans un tableau Rich."""
    console = Console()
    table = Table(title="Métriques d'évaluation", show_header=True, header_style="bold cyan")
    table.add_column("Métrique", style="bold")
    table.add_column("Valeur", justify="right")
    table.add_column("Interprétation")

    table.add_row("Brier Score",       f"{metrics['brier_score']:.4f}",       "< 0.25 = bat le naïf")
    table.add_row("Brier Skill Score", f"{metrics['brier_skill_score']:.4f}",  "> 0 = meilleur que P=0.5")
    table.add_row("Log Loss",          f"{metrics['log_loss']:.4f}",           "< 0.693 = bat le naïf")
    table.add_row("AUC-ROC",           f"{metrics['auc_roc']:.4f}",            "> 0.5 = discriminant")
    table.add_row("ECE",               f"{metrics['ece']:.4f}",                "< 0.05 = bien calibré")

    console.print(table)
