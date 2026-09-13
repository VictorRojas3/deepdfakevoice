import base64
import logging
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from schemas import DetectResponse
import features as FE

log = logging.getLogger("detect")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "detector.joblib"))
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))

app = FastAPI(title="Altur — Detector de Bot", version="1.0")
STATE = {}

@app.on_event("startup")
def _load():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"No existe {MODEL_PATH}. Ejecuta el entrenamiento primero.")
    
    b = joblib.load(MODEL_PATH)
    STATE.update(
        detector=b["detector"],
        columns=b["timing_keys"] + b["acoustic_keys"] + FE.LEVEL_KEYS
    )
    log.info("Modelo principal cargado (%s)", MODEL_PATH.name)

async def _extract_audio(request: Request) -> bytes:
    ctype = (request.headers.get("content-type") or "").lower()
    body = await request.body()
    
    if not body:
        raise HTTPException(400, "Body vacío")

    if ctype.startswith("application/json"):
        import json
        try:
            payload = json.loads(body)
            b64_str = payload.get("audio") or payload.get("audio_base64")
            if not b64_str:
                raise HTTPException(400, "Se requiere el campo 'audio' con el base64")
            return base64.b64decode(b64_str)
        except Exception as e:
            raise HTTPException(400, f"JSON o base64 inválido: {e}")
    return body

@app.post("/detect", response_model=DetectResponse)
async def detect(request: Request):
    wav = await _extract_audio(request)
    try:
        f = FE.decompose(wav)
    except Exception as e:
        raise HTTPException(400, f"Error procesando audio: {e}")
    try:
        X = pd.DataFrame([[f[c] for c in STATE["columns"]]], columns=STATE["columns"])
    except KeyError as e:
        raise HTTPException(500, f"Faltan features generadas: {e}")
    p_synthetic = float(STATE["detector"].predict_proba(X)[0])
    is_syn = bool(p_synthetic >= DECISION_THRESHOLD)    
    conf = p_synthetic if is_syn else 1.0 - p_synthetic
    log.info("detect -> is_synthetic=%s confidence=%.4f", is_syn, conf)
    return DetectResponse(is_synthetic=is_syn, confidence=round(conf, 4))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "detector" in STATE}