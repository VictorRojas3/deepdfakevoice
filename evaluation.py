"""Evaluación probabilística: métricas, calibración, coverage-risk y significancia.

El criterio de selección NO es accuracy. Entre modelos con accuracy comparable se
prefiere el de menor Brier/log-loss, mejor ECE, mayor precisión a alta confianza y
menor latencia. Este módulo produce todo eso con intervalos de confianza, porque con
n=71 en el holdout oficial muchas de estas cantidades tienen error de estimación
del mismo orden que las diferencias entre modelos.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)

EPS = 1e-12


def ece(y, p, bins=10, strategy="uniform"):
    """Expected Calibration Error: E|confianza - frecuencia real|, ponderado por bin."""
    y, p = np.asarray(y), np.asarray(p)
    edges = (np.linspace(0, 1, bins + 1) if strategy == "uniform"
             else np.quantile(p, np.linspace(0, 1, bins + 1)))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    total = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def mce(y, p, bins=10):
    """Maximum Calibration Error: el peor bin, no el promedio."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    worst = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() >= 3:                       # bins con <3 casos no son informativos
            worst = max(worst, abs(p[m].mean() - y[m].mean()))
    return float(worst)


def reliability_table(y, p, bins=10):
    """Diagrama de fiabilidad en forma de tabla: si el modelo dice 0.9, ¿es 90%?"""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append({
            "bin": f"[{edges[b]:.1f},{edges[b+1]:.1f})",
            "n": int(m.sum()),
            "p_media": float(p[m].mean()) if m.sum() else np.nan,
            "frec_real": float(y[m].mean()) if m.sum() else np.nan,
            "gap": float(p[m].mean() - y[m].mean()) if m.sum() else np.nan,
        })
    return rows


def confidence_of(p, threshold=0.5):
    """Confianza en la clase AFIRMADA. Con threshold=0.5 coincide con max(p,1-p);
    con otro umbral no, y por eso se deriva de la decisión y no al revés."""
    p = np.asarray(p)
    return np.where(p >= threshold, p, 1.0 - p)


def coverage_risk(y, p, threshold=0.5, grid=None):
    """Curva cobertura-riesgo: si solo decidimos cuando hay confianza >= c,
    ¿qué fracción cubrimos y qué error cometemos condicionado a haber decidido?"""
    y, p = np.asarray(y), np.asarray(p)
    conf = confidence_of(p, threshold)
    pred = (p >= threshold).astype(int)
    grid = np.arange(0.50, 1.00, 0.01) if grid is None else np.asarray(grid)
    out = []
    for c in grid:
        m = conf >= c
        n = int(m.sum())
        out.append({
            "conf_min": float(c), "coverage": float(m.mean()), "n": n,
            "accuracy": float(accuracy_score(y[m], pred[m])) if n else np.nan,
            "risk": float(1 - accuracy_score(y[m], pred[m])) if n else np.nan,
            "precision": (float(precision_score(y[m], pred[m], zero_division=0))
                          if n and pred[m].sum() else np.nan),
        })
    return out


def precision_at_confidence(y, p, levels=(0.90, 0.95, 0.99), threshold=0.5, n_boot=2000,
                            seed=0):
    """Precisión y cobertura a confianza alta, CON intervalo de confianza.

    Con n=71 estos números tienen error de estimación grande; reportarlos sin IC
    invita a leer diferencias que son ruido. `estimable` marca cuándo el soporte
    es demasiado pequeño para concluir nada.
    """
    y, p = np.asarray(y), np.asarray(p)
    conf = confidence_of(p, threshold)
    pred = (p >= threshold).astype(int)
    rng = np.random.default_rng(seed)
    out = []
    for c in levels:
        m = conf >= c
        n, npos = int(m.sum()), int(pred[m].sum())
        if npos == 0:
            out.append({"conf": c, "coverage": float(m.mean()), "n": n,
                        "precision": np.nan, "ci": (np.nan, np.nan), "estimable": False})
            continue
        prec = precision_score(y[m], pred[m], zero_division=0)
        yb, pb = y[m], pred[m]
        boot = []
        for _ in range(n_boot):
            i = rng.integers(0, n, n)
            if pb[i].sum():
                boot.append(precision_score(yb[i], pb[i], zero_division=0))
        lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan))
        out.append({"conf": c, "coverage": float(m.mean()), "n": n,
                    "precision": float(prec), "ci": (float(lo), float(hi)),
                    "estimable": npos >= 20})
    return out


def evaluate(y, p, threshold=0.5, n_boot=2000, seed=0, latency_ms=None):
    """Todas las métricas del brief en un solo dict, con IC bootstrap para AUC/acc."""
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    pc = np.clip(p, EPS, 1 - EPS)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    rng = np.random.default_rng(seed)
    b_auc, b_acc, b_bri = [], [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        b_auc.append(roc_auc_score(y[i], p[i]))
        b_acc.append(accuracy_score(y[i], pred[i]))
        b_bri.append(brier_score_loss(y[i], p[i]))

    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, pc)),
        "ece": ece(y, p), "mce": mce(y, p),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "auc_ci": (float(np.percentile(b_auc, 2.5)), float(np.percentile(b_auc, 97.5))) if b_auc else (np.nan,) * 2,
        "acc_ci": (float(np.percentile(b_acc, 2.5)), float(np.percentile(b_acc, 97.5))) if b_acc else (np.nan,) * 2,
        "brier_ci": (float(np.percentile(b_bri, 2.5)), float(np.percentile(b_bri, 97.5))) if b_bri else (np.nan,) * 2,
        "latency_ms": float(latency_ms) if latency_ms is not None else None,
    }


# ------------------------------------------------------------------ significancia
def delong_or_bootstrap_diff(y, p_a, p_b, metric="roc_auc", n_boot=4000, seed=0):
    """¿La diferencia entre dos modelos es distinguible del ruido? Bootstrap pareado.

    Remuestrea las MISMAS llamadas para ambos modelos, así la comparación no se
    contamina con la varianza compartida de la muestra.
    """
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    fn = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score,
          "brier": brier_score_loss}[metric]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        diffs.append(fn(y[i], p_a[i]) - fn(y[i], p_b[i]))
    diffs = np.array(diffs)
    return {"diff": float(fn(y, p_a) - fn(y, p_b)),
            "ci": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
            "p_two_sided": float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))),
            "degenerado": bool(np.allclose(diffs, 0)),   # sin resolución: no concluir nada
            "significativo": bool(np.percentile(diffs, 2.5) > 0 or np.percentile(diffs, 97.5) < 0)}


def error_correlation(y, p_a, p_b, threshold=0.5):
    """¿Dos ramas fallan en las mismas llamadas? Si sí, no necesitamos ambas."""
    y = np.asarray(y)
    ea = ((np.asarray(p_a) >= threshold).astype(int) != y).astype(int)
    eb = ((np.asarray(p_b) >= threshold).astype(int) != y).astype(int)
    both = int(((ea == 1) & (eb == 1)).sum())
    return {
        "errores_a": int(ea.sum()), "errores_b": int(eb.sum()), "errores_comunes": both,
        "jaccard": float(both / max(((ea == 1) | (eb == 1)).sum(), 1)),
        "phi": float(np.corrcoef(ea, eb)[0, 1]) if ea.std() and eb.std() else np.nan,
        "corr_residuos": float(np.corrcoef(y - np.asarray(p_a), y - np.asarray(p_b))[0, 1]),
    }

