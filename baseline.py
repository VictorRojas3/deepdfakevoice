"""Baseline de un solo umbral, DERIVADO — no ajustado.

La regla completa cabe en una línea:

    de cada llamada, mide qué fracción de las respuestas del caller tardó más de
    TAU = 2 s. Si esa fracción supera q*, es un bot.

Lo interesante es que q* no se escoge a ojo ni por fuerza bruta: sale de Neyman-Pearson.

    Modelo: cada intercambio es una Bernoulli con P(lenta) = p.
            p_H = P(lambda > TAU | humano),  p_S = P(lambda > TAU | bot).

    log LR(K, r) = K log(p_S/p_H) + (r-K) log((1-p_S)/(1-p_H))

    decidir bot  <=>  log LR > 0  <=>  K/r > q* = -b/(a-b),
    con a = log(p_S/p_H) y b = log((1-p_S)/(1-p_H)).

Y el umbral derivado generaliza MEJOR que el ajustado por fuerza bruta sobre train
(0.9014 vs 0.8873 en val): ajustar sobreajusta, deducir no.

Por qué funciona, en una frase: un solo intercambio casi no informa (d = 0.39,
AUC 0.675), pero una llamada da ~10 oportunidades de medirlo y promediar concentra.

Uso:
    python baseline.py          deriva, evalúa y guarda baseline.joblib
"""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.metrics import accuracy_score, roc_auc_score
import features as FE
from detector import ThresholdBaseline, derive_threshold, posterior

ROOT = Path(__file__).resolve().parent
DATA = next((d for d in (ROOT, ROOT.parent) if (d / "manifest.csv").exists()), ROOT)
OUT = ROOT / "baseline.joblib"
POS = "synthetic"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=FE.TAU_SLOW)
    args = ap.parse_args()
    man = pd.read_csv(DATA / "manifest.csv")
    lam, y, split = [], [], []
    for r in man.itertuples():
        L, _ = FE.call_latencies(DATA / "audio" / f"{r.anon_id}.wav")
        lam.append(L)
        y.append(1 if r.label == POS else 0)
        split.append(r.split)
    y = np.array(y)
    split = np.array(split)
    tr, va = split == "train", split == "val"
    m = ThresholdBaseline(tau=args.tau).fit([lam[i] for i in np.where(tr)[0]], y[tr])
    x = np.array([(L > m.tau).mean() if len(L) else 0.0 for L in lam])
    cands = np.unique(x[tr])
    cands = (cands[:-1] + cands[1:]) / 2
    q_emp = cands[int(np.argmax([accuracy_score(y[tr], (x[tr] > c).astype(int)) for c in cands]))]
    for nm, q in (("DERIVADO", m.q_), ("ajustado en train", q_emp)):
        pred = (x[va] > q).astype(int)
        tn, fp = int(((pred == 0) & (y[va] == 0)).sum()), int(((pred == 1) & (y[va] == 0)).sum())
        fn, tp = int(((pred == 0) & (y[va] == 1)).sum()), int(((pred == 1) & (y[va] == 1)).sum())
    ph = np.concatenate([lam[i] for i in np.where(y == 0)[0] if len(lam[i])])
    ps = np.concatenate([lam[i] for i in np.where(y == 1)[0] if len(lam[i])])
    sp = np.sqrt((ph.var() * len(ph) + ps.var() * len(ps)) / (len(ph) + len(ps)))
    d1 = (ps.mean() - ph.mean()) / sp
    mh = np.array([L.mean() for i, L in enumerate(lam) if y[i] == 0 and len(L)])
    ms = np.array([L.mean() for i, L in enumerate(lam) if y[i] == 1 and len(L)])
    dc = (ms.mean() - mh.mean()) / np.sqrt(
        (mh.var() * len(mh) + ms.var() * len(ms)) / (len(mh) + len(ms)))
    rbar = np.mean([len(L) for L in lam if len(L)])
    r_ef = (dc / d1) ** 2
    for q in (5, 25, 50, 75, 90, 95):
        a_, b_ = np.percentile(ph, q), np.percentile(ps, q)
    for k in (1, 2, 5, 7):
        joblib.dump({"baseline": m, "tau": m.tau, "p_h": m.p_h_, "p_s": m.p_s_,
                 "q_derived": m.q_, "q_bruteforce": float(q_emp), "shrink": m.shrink_,
                 "val_acc": float(accuracy_score(y[va], (x[va] > m.q_).astype(int))),
                 "val_auc": float(roc_auc_score(y[va], x[va]))}, OUT)
    from sklearn.metrics import brier_score_loss
    pv = np.array([m.score(lam[i]) for i in np.where(va)[0]])
    pv0 = np.array([posterior(int((lam[i] > m.tau).sum()), len(lam[i]), m.a_, m.b_)
                    if len(lam[i]) else 0.5 for i in np.where(va)[0]])
if __name__ == "__main__":
    main()
