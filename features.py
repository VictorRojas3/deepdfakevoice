"""Descomposición de una llamada completa en features para detectar caller sintético.

Entrada: matriz X de forma (N, 2), N muestras a 8000 por segundo.
    columna 0 = caller (lo que hay que clasificar)
    columna 1 = agente (referencia: sin ella no existe la latencia)
Salida: vector v en R^57.

El nivel de grabación del caller filtra la etiqueta (~5 dB entre clases). Se normaliza
para que no se cuele por la puerta de atrás vía el VAD, y queda expuesto como la feature
`caller_level_db`, donde su peso es visible y se puede apagar.
"""
import io
import numpy as np
import soundfile as sf

SR_TARGET = 8000
FRAME = 0.02          # ventana de 20 ms -> 160 muestras a 8 kHz
PAD = 0.08            # margen a cada lado de un intervalo detectado
MERGE = 0.20          # huecos menores a esto no parten el turno
MIN_DUR = 0.20        # descarta intervalos espurios
THR_DB = -40.0          # suelo absoluto, sobre señal normalizada a máximo
NOISE_MARGIN_DB = 12.0  # umbral = piso de ruido del canal + esto (ver _threshold)
N_MFCC = 13
TAU_SLOW = 2.0        # umbral interno: una respuesta es 'lenta' si tarda más de esto

FEATURE_SCHEMA_VERSION = "2.1"   # 2.1: umbral de VAD adaptativo al piso de ruido

TIMING_KEYS = [
    'r0', 'r1', 'n0', 'n1', 'm0', 's0', 'm1', 's1',
    'lat_mean', 'lat_med', 'lat_std', 'lat_p10', 'lat_p90',
    'lat_slow', 'lat_p75', 'lat_p25', 'lat_iqr', 'lat_fast', 'lat_neg',
    'lat_skew', 'lat_kurt', 'lat_dur_corr', 'cvlat', 'cv0',
    'bargein', 'bargein_rate', 'silence', 'silence_true', 'overlap_time', 'duration_s',
]
ACOUSTIC_KEYS = ([f'mf_mu_{i}' for i in range(N_MFCC)]
                 + [f'mf_sd_{i}' for i in range(N_MFCC)])
LEVEL_KEYS = ['caller_level_db']
ALL_KEYS = TIMING_KEYS + ACOUSTIC_KEYS + LEVEL_KEYS


def _frame_energy_db(sig, sr):
    """Parte la señal en bloques de 20 ms y devuelve la norma L2 de cada bloque, en escala log.

    Reduce una secuencia de ~1.2M números a una de ~7500. Una sola pasada vectorizada.
    """
    n = int(FRAME * sr)
    nf = len(sig) // n
    if nf == 0:
        return np.zeros(0, dtype=np.float32)
    return 20 * np.log10(np.sqrt((sig[:nf * n].reshape(nf, n) ** 2).mean(1)) + 1e-10)


def _threshold(e_db, margin=None, floor_db=THR_DB):
    """Umbral de VAD relativo al piso de ruido DE ESE CANAL.

    Un umbral fijo de -40 dB funciona en audio telefónico (silencio digital, piso
    ~-72 dB) y falla por completo en grabaciones de micrófono en sala (piso ~-29 dB):
    ahí marca el 100% de los frames como habla, no quedan transiciones y la rama de
    timing se queda ciega. Medido sobre un dataset externo: r pasó de 10.4 a 0.0 y
    las 12 llamadas humanas se clasificaron como sintéticas.

    Estimar el piso como el percentil 10 del propio canal y sumar un margen hace el
    VAD invariante a la condición de grabación. El suelo absoluto evita que un canal
    casi todo silencio (piso ~-200 dB, PCM digital) baje el umbral a la nada.
    """
    if len(e_db) == 0:
        return floor_db
    margin = NOISE_MARGIN_DB if margin is None else margin
    noise = float(np.percentile(e_db, 10))
    if noise < -90:                     
        return floor_db
    return max(noise + margin, floor_db)


def _segments(e_db, thr_db):
    """Umbraliza y agrupa: secuencia de energías -> conjunto de intervalos disjuntos.

    Aquí acaba la señal y empieza la combinatoria. Todo lo que sigue es aritmética
    sobre intervalos.
    """
    act = e_db > thr_db
    out = []
    i, nf = 0, len(act)
    while i < nf:
        if not act[i]:
            i += 1
            continue
        j = i
        while j < nf and act[j]:
            j += 1
        start, end = max(0.0, i * FRAME - PAD), j * FRAME + PAD
        if out and start - out[-1][1] <= MERGE:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
        i = j
    return [(a, b) for a, b in out if b - a >= MIN_DUR]


def _skew(x):
    if len(x) < 3 or x.std() == 0:
        return 0.0
    return float((((x - x.mean()) / x.std()) ** 3).mean())


def _kurt(x):
    if len(x) < 4 or x.std() == 0:
        return 0.0
    return float((((x - x.mean()) / x.std()) ** 4).mean() - 3.0)


def _safe_corr(a, b):
    if len(a) < 3 or len(a) != len(b) or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _union_measure(intervals):
    """Medida de Lebesgue de la unión de intervalos (maneja solapes)."""
    if not intervals:
        return 0.0
    total, cur_a, cur_b = 0.0, None, None
    for a, b in sorted(intervals):
        if cur_a is None:
            cur_a, cur_b = a, b
        elif a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            total += cur_b - cur_a
            cur_a, cur_b = a, b
    return total + (cur_b - cur_a if cur_a is not None else 0.0)


def _overlap_measure(c0, c1):
    """Medida del conjunto donde AMBOS hablan: |A| + |B| - |A u B|."""
    ma = sum(b - a for a, b in c0)
    mb = sum(b - a for a, b in c1)
    return max(0.0, ma + mb - _union_measure(c0 + c1))


def latencies(c0, c1):
    """El multiconjunto Lambda: la cantidad central de toda la solución.

    Ordena I0 u I1 por extremo izquierdo y, para cada par consecutivo (agente, caller),
    mide  lambda = (inicio del caller) - (fin del agente).  lambda < 0 <=> se solapan.

    Devuelve (Lambda, n_solapes, duraciones_del_turno_previo_del_agente).

    Lo medido sobre las 353 llamadas (3689 valores de lambda):

        P(lambda < 0):  humano 0.398  vs  sintético 0.342   -> NO separa
        P(lambda > 2):  humano 0.058  vs  sintético 0.447   -> 7.7x, AQUÍ separa

    Es decir: los dos interrumpen por igual. La firma no está en interrumpir, sino en
    cuánto se tarda cuando NO se interrumpe — el bot paga ASR + LLM + TTS y el humano no.
    Los cuantiles bajos de las dos clases coinciden; divergen a partir de la mediana.
    """
    t = sorted([(a, b, 0) for a, b in c0] + [(a, b, 1) for a, b in c1])
    lat, overlap, prev_dur = [], 0, []
    for cur, nxt in zip(t, t[1:]):
        if cur[2] == 1 and nxt[2] == 0:         
            lat.append(nxt[0] - cur[1])
            prev_dur.append(cur[1] - cur[0])    
            if nxt[0] < cur[1]:
                overlap += 1
    if not lat:
        return np.array([]), 0, np.array([])
    return np.array(lat), overlap, np.array(prev_dur)


def _timing(c0, c1, dur):
    """Estadísticos de los dos conjuntos de intervalos. Requiere ambos: la latencia
    es una diferencia entre un extremo de c1 y el siguiente extremo de c0."""
    d0 = np.array([b - a for a, b in c0]) if c0 else np.array([0.0])
    d1 = np.array([b - a for a, b in c1]) if c1 else np.array([0.0])
    lat, bargein, prev_agent_dur = latencies(c0, c1)
    if len(lat) == 0:
        lat, prev_agent_dur = np.array([0.0]), np.array([0.0])
    return {
        'r0': d0.sum() / dur, 'r1': d1.sum() / dur,
        'n0': float(len(c0)), 'n1': float(len(c1)),
        'm0': d0.mean(), 's0': d0.std(), 'm1': d1.mean(), 's1': d1.std(),
        'lat_mean': lat.mean(), 'lat_med': float(np.median(lat)), 'lat_std': lat.std(),
        'lat_p10': float(np.percentile(lat, 10)), 'lat_p90': float(np.percentile(lat, 90)),        
        'lat_slow': float((lat > TAU_SLOW).mean()),        
        'lat_p75': float(np.percentile(lat, 75)),
        'lat_p25': float(np.percentile(lat, 25)),
        'lat_iqr': float(np.percentile(lat, 75) - np.percentile(lat, 25)),
        'lat_fast': float((lat < 0.3).mean()),
        'lat_neg': float((lat < 0).mean()),
        'lat_skew': float(_skew(lat)),
        'lat_kurt': float(_kurt(lat)),
        'lat_dur_corr': float(_safe_corr(lat, prev_agent_dur)),
        'cvlat': lat.std() / (abs(lat.mean()) + 1e-9),
        'cv0': d0.std() / (d0.mean() + 1e-9),
        'bargein': float(bargein), 'bargein_rate': bargein / max(len(c0), 1),
        'silence': dur - (d0.sum() + d1.sum()),
        'silence_true': dur - _union_measure(c0 + c1),
        'overlap_time': _overlap_measure(c0, c1),
    }


def _acoustic(sig, sr, segs):
    """Descriptor de 26 números del caller, calculado SOLO sobre los intervalos de habla.

    Sobre la secuencia completa el descriptor mide la proporción de silencio, no la voz:
    medido, corr(descriptor, proporción de habla) = 0.92 sobre todo el canal contra 0.12
    restringido a los intervalos.
    """
    import librosa
    zeros = {k: 0.0 for k in ACOUSTIC_KEYS}
    if not segs:
        return zeros
    speech = np.concatenate([sig[int(a * sr):int(b * sr)] for a, b in segs])
    if len(speech) < int(0.1 * sr):
        return zeros
    m = librosa.feature.mfcc(y=speech, sr=sr, n_mfcc=N_MFCC)   # (13, n_ventanas)
    return {**{f'mf_mu_{i}': float(v) for i, v in enumerate(m.mean(1))},
            **{f'mf_sd_{i}': float(v) for i, v in enumerate(m.std(1))}}


def load_call(source):
    """Devuelve (caller, agent, sr, degraded). Acepta ruta, bytes o file-like."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    x, sr = sf.read(source, dtype='float32', always_2d=True)

    degraded = None
    if x.shape[1] == 1:
        # Sin columna del agente no hay latencia: modo degradado, solo descriptor.
        caller, agent = x[:, 0], np.zeros_like(x[:, 0])
        degraded = 'mono: sin canal de agente, features de timing no disponibles'
    else:
        caller, agent = x[:, 0], x[:, 1]     # contrato del reto: 0=caller, 1=agente

    if sr != SR_TARGET:
        import librosa
        caller = librosa.resample(caller, orig_sr=sr, target_sr=SR_TARGET)
        agent = librosa.resample(agent, orig_sr=sr, target_sr=SR_TARGET)
        sr = SR_TARGET
    return caller, agent, sr, degraded


def decompose(source, normalize_level=True):
    """source: ruta, bytes o file-like de un WAV. Devuelve dict con ALL_KEYS."""
    caller, agent, sr, degraded = load_call(source)
    dur = max(len(caller) / sr, 1e-6)

    peak = float(np.abs(caller).max())
    caller_n = caller / (peak + 1e-9) if normalize_level else caller
    agent_peak = float(np.abs(agent).max())
    agent_n = agent / (agent_peak + 1e-9)

    e0 = _frame_energy_db(caller_n, sr)
    e1 = _frame_energy_db(agent_n, sr) if agent_peak > 0 else np.zeros(0, dtype=np.float32)

    c0 = _segments(e0, _threshold(e0))
    c1 = _segments(e1, _threshold(e1))

    f = _timing(c0, c1, dur)
    f['duration_s'] = dur
    f['caller_level_db'] = 20 * np.log10(peak + 1e-9)    # el confound, declarado
    f.update(_acoustic(caller_n, sr, c0))

    f['_degraded'] = degraded
    f['_n_caller_segments'] = len(c0)
    f['_n_agent_segments'] = len(c1)
    return f


def call_latencies(source):
    """WAV -> (Lambda, r). Atajo para el baseline y para análisis."""
    caller, agent, sr, _ = load_call(source)
    cn = caller / (np.abs(caller).max() + 1e-9)
    ap = float(np.abs(agent).max())
    an = agent / (ap + 1e-9)
    e0, e1 = _frame_energy_db(cn, sr), _frame_energy_db(an, sr)
    c0 = _segments(e0, _threshold(e0))
    c1 = _segments(e1, _threshold(e1)) if ap > 0 else []
    lat, _, _ = latencies(c0, c1)
    return lat, len(lat)


def to_vector(f, keys):
    """dict de features -> np.array en el orden dado (contrato con el modelo)."""
    return np.array([f[k] for k in keys], dtype=np.float64)
