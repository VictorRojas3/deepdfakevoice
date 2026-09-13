import argparse
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

import features as FE
from detector import LateFusionDetector

ROOT = Path(__file__).resolve().parent
DATA = next((d for d in (ROOT, ROOT.parent) if (d / "manifest.csv").exists()), ROOT)
CACHE = ROOT / "features_cache.csv"
MODEL_OUT = ROOT / "detector.joblib"
POS = "synthetic"

def build_dataset(use_cache=True):
    if use_cache and CACHE.exists():
        return pd.read_csv(CACHE)

    man = pd.read_csv(DATA / "manifest.csv")
    rows = []
    t0 = time.perf_counter()
    
    for i, r in enumerate(man.itertuples(), 1):
        f = FE.decompose(DATA / "audio" / f"{r.anon_id}.wav")
        f = {k: v for k, v in f.items() if not k.startswith('_')}
        f.update(anon_id=r.anon_id, label=r.label, split=r.split)
        rows.append(f)
        if i % 100 == 0:
            print(f"  {i}/{len(man)} procesadas...")
            
    df = pd.DataFrame(rows)
    df.to_csv(CACHE, index=False)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true", help="Forzar extracción de audios")
    ap.add_argument("--calibration", default="sigmoid", choices=["sigmoid", "isotonic", "none"])
    args = ap.parse_args()

    df = build_dataset(use_cache=not args.no_cache)
    cal = None if args.calibration == "none" else args.calibration

    tr = df[df.split == "train"]
    va = df[df.split == "val"]
    y_tr = (tr.label == POS).astype(int).values
    y_va = (va.label == POS).astype(int).values
    
    prod = LateFusionDetector(FE.TIMING_KEYS, FE.ACOUSTIC_KEYS, calibration=cal).fit(tr, y_tr)
    
    p_va = prod.predict_proba(va)
    y_pred_va = (p_va >= 0.5).astype(int)
    val_bacc = balanced_accuracy_score(y_va, y_pred_va)
    val_acc = accuracy_score(y_va, y_pred_va)
    val_auc = roc_auc_score(y_va, p_va)
    p_tr = prod.predict_proba(tr)
    y_pred_tr = (p_tr >= 0.5).astype(int)
    tr_bacc = balanced_accuracy_score(y_tr, y_pred_tr)
    joblib.dump({
        "detector": prod,
        "timing_keys": FE.TIMING_KEYS,
        "acoustic_keys": FE.ACOUSTIC_KEYS,
        "threshold": 0.5,
        "val_metrics": {
            "balanced_accuracy": float(val_bacc),
            "accuracy": float(val_acc),
            "auc": float(val_auc)
        },
        "sklearn_version": __import__("sklearn").__version__,
    }, MODEL_OUT)    
    return 0

if __name__ == "__main__":
    sys.exit(main())