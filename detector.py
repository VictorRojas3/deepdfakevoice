"""Arquitecturas del detector.

    ThresholdBaseline   — la regla de un umbral, derivada por Neyman-Pearson.
    LateFusionDetector  — dos modelos base + meta + calibración.

Ambas viven aquí, y no dentro de los scripts que las entrenan, para que los objetos
serializados se puedan deserializar desde cualquier proceso: si una clase se define en
el script que se ejecuta, pickle la guarda como `__main__.Clase` y la API no la carga.
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
C_BASE = 1.0
FOLDS = 5


def make_base(C=C_BASE, seed=SEED):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000, C=C, random_state=seed))


class LateFusionDetector:
    """timing   -> p_t ┐
                       ├─ meta-logreg -> calibración -> P(synthetic)
       acústica -> p_a ┘

    Motivo: una logística con penalización L2 reparte UN solo presupuesto ||w||^2 entre
    las 47 coordenadas. El bloque acústico (26 dims correlacionadas) se lo come y los
    coeficientes de timing quedan encogidos. Barrer C mueve el presupuesto total pero no
    puede repartirlo distinto entre bloques: un escalar no desacopla dos bloques. Ajustar
    cada bloque por separado y recombinar equivale a darle a cada uno su propia
    penalización, y añade 2 parámetros que reescalan sus aportes.

    Las predicciones que alimentan al meta son out-of-fold: cada p_i viene de un modelo
    que no vio x_i. Con predicciones in-sample el meta observaría bases artificialmente
    buenas, aprendería a confiar ciegamente en ellas, y sus pesos serían los equivocados
    para inferencia.
    """

    def __init__(self, timing_keys, acoustic_keys, calibration="sigmoid",
                 C=C_BASE, folds=FOLDS, seed=SEED):
        self.timing_keys = list(timing_keys)
        self.acoustic_keys = list(acoustic_keys)
        self.calibration = calibration
        self.C = C
        self.folds = folds
        self.seed = seed

    def fit(self, X, y):
        cv = StratifiedKFold(self.folds, shuffle=True, random_state=self.seed)
        Xt, Xa = X[self.timing_keys].values, X[self.acoustic_keys].values

        self.oof_ = np.column_stack([
            cross_val_predict(make_base(self.C, self.seed), Xt, y, cv=cv,
                              method="predict_proba")[:, 1],
            cross_val_predict(make_base(self.C, self.seed), Xa, y, cv=cv,
                              method="predict_proba")[:, 1],
        ])

        self.m_timing_ = make_base(self.C, self.seed).fit(Xt, y)
        self.m_acoustic_ = make_base(self.C, self.seed).fit(Xa, y)

        meta = LogisticRegression(max_iter=5000, random_state=self.seed)
        if self.calibration in ("sigmoid", "isotonic"):
            self.meta_ = CalibratedClassifierCV(
                meta, method=self.calibration,
                cv=StratifiedKFold(self.folds, shuffle=True, random_state=self.seed)
            ).fit(self.oof_, y)
        else:
            self.meta_ = meta.fit(self.oof_, y)
        return self

    def base_scores(self, X):
        return np.column_stack([
            self.m_timing_.predict_proba(X[self.timing_keys].values)[:, 1],
            self.m_acoustic_.predict_proba(X[self.acoustic_keys].values)[:, 1],
        ])

    def predict_proba(self, X):
        return self.meta_.predict_proba(self.base_scores(X))[:, 1]


def derive_threshold(p_h, p_s):
    """Umbral óptimo de Neyman-Pearson para el modelo Bernoulli. Devuelve (q*, a, b)."""
    a = np.log(p_s / p_h)
    b = np.log((1 - p_s) / (1 - p_h))
    return -b / (a - b), a, b


def posterior(k, r, a, b, shrink=1.0):
    """P(bot | K=k, r) con prior 1/2. Es LR/(1+LR), estable en log.

    `shrink` corrige la independencia que el modelo Bernoulli supone y que NO se cumple:
    las lambda dentro de una llamada están correlacionadas (rho ~ 0.11), así que el
    tamaño de muestra efectivo es r_ef ~ 0.49 r. Sin corregir, la razón de verosimilitud
    cuenta cada intercambio como evidencia nueva y devuelve confianzas de 1.0000.
    Escalar el log-LR por r_ef/r no cambia ninguna decisión (el signo se preserva),
    solo deja de exagerar la certeza.
    """
    llr = shrink * (a * k + b * (r - k))
    return float(1.0 / (1.0 + np.exp(-np.clip(llr, -30, 30))))


class ThresholdBaseline:
    """Dos números estimados de miles de intercambios. No hay nada que sobreajustar."""

    def __init__(self, tau=2.0):
        self.tau = tau

    def fit(self, lam_list, y):
        """lam_list: lista de arrays Lambda (uno por llamada). y: 1 = sintético."""
        pooled_h = np.concatenate([L for L, t in zip(lam_list, y) if t == 0 and len(L)])
        pooled_s = np.concatenate([L for L, t in zip(lam_list, y) if t == 1 and len(L)])
        self.p_h_ = float((pooled_h > self.tau).mean())
        self.p_s_ = float((pooled_s > self.tau).mean())
        self.q_, self.a_, self.b_ = derive_threshold(self.p_h_, self.p_s_)
        self.n_h_, self.n_s_ = len(pooled_h), len(pooled_s)
        self.shrink_ = self._effective_ratio(lam_list, y)
        return self

    @staticmethod
    def _effective_ratio(lam_list, y):
        """r_ef / r, estimado comparando el tamaño de efecto de UNA lambda contra el de
        la media por llamada. Si fueran independientes la ganancia sería sqrt(r)."""
        h = [L for L, t in zip(lam_list, y) if t == 0 and len(L)]
        s = [L for L, t in zip(lam_list, y) if t == 1 and len(L)]
        if not h or not s:
            return 1.0
        ph, ps = np.concatenate(h), np.concatenate(s)
        sp = np.sqrt((ph.var() * len(ph) + ps.var() * len(ps)) / (len(ph) + len(ps)))
        if sp == 0:
            return 1.0
        d1 = (ps.mean() - ph.mean()) / sp
        mh = np.array([L.mean() for L in h])
        ms = np.array([L.mean() for L in s])
        spc = np.sqrt((mh.var() * len(mh) + ms.var() * len(ms)) / (len(mh) + len(ms)))
        if spc == 0 or d1 == 0:
            return 1.0
        dc = (ms.mean() - mh.mean()) / spc
        rbar = np.mean([len(L) for L in h + s])
        return float(np.clip(((dc / d1) ** 2) / rbar, 0.05, 1.0))

    def score(self, L):
        """Lambda de una llamada -> P(sintético). Sin latencias, no hay evidencia."""
        r = len(L)
        if r == 0:
            return 0.5
        return posterior(int((np.asarray(L) > self.tau).sum()), r,
                         self.a_, self.b_, getattr(self, "shrink_", 1.0))

    def predict(self, L):
        r = len(L)
        if r == 0:
            return False
        return bool((np.asarray(L) > self.tau).mean() > self.q_)




# =============================================================================
#  Fusión de N ramas
# =============================================================================
def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


class MultiBranchFusion:
    """Fusiona K ramas probabilísticas en un único P(synthetic|X).

    Tres modos, comparables entre sí:

      'stacking'  meta-logreg sobre las probabilidades crudas
      'logodds'   logit(p_final) = b + sum_k w_k logit(p_k)   <- forma bayesiana:
                  sumar log-odds equivale a multiplicar razones de verosimilitud,
                  pero con pesos APRENDIDOS en vez de w_k = 1. Los pesos aprendidos
                  son precisamente la corrección por la NO independencia entre ramas:
                  si dos ramas ven lo mismo, el meta les baja el peso conjunto.
      'mean'      promedio simple de probabilidades (línea base sin parámetros)

    Las entradas del meta son out-of-fold en `fit`: cada p_k(x_i) viene de un modelo
    que no vio x_i. Con predicciones in-sample el meta observa ramas artificialmente
    buenas y aprende pesos que no valen en inferencia.
    """

    def __init__(self, mode="logodds", calibration="sigmoid", folds=FOLDS, seed=SEED):
        self.mode, self.calibration, self.folds, self.seed = mode, calibration, folds, seed

    def fit(self, oof_scores, y):
        """oof_scores: (n, K) probabilidades out-of-fold de cada rama."""
        Z = np.asarray(oof_scores, dtype=float)
        self.k_ = Z.shape[1]
        if self.mode == "mean":
            self.meta_ = None
            return self
        Xm = _logit(Z) if self.mode == "logodds" else Z
        base = LogisticRegression(max_iter=5000, random_state=self.seed)
        if self.calibration in ("sigmoid", "isotonic"):
            self.meta_ = CalibratedClassifierCV(
                base, method=self.calibration,
                cv=StratifiedKFold(self.folds, shuffle=True, random_state=self.seed)).fit(Xm, y)
        else:
            self.meta_ = base.fit(Xm, y)
        return self

    def predict_proba(self, scores):
        Z = np.asarray(scores, dtype=float)
        if self.mode == "mean":
            return Z.mean(axis=1)
        Xm = _logit(Z) if self.mode == "logodds" else Z
        return self.meta_.predict_proba(Xm)[:, 1]

    def weights(self):
        """Pesos aprendidos por rama. Un peso ~0 dice que esa rama es redundante."""
        if self.meta_ is None:
            return np.full(self.k_, 1.0 / self.k_)
        m = self.meta_
        est = (m.calibrated_classifiers_[0].estimator
               if hasattr(m, "calibrated_classifiers_") else m)
        return np.asarray(est.coef_[0], dtype=float)
