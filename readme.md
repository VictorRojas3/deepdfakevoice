Synthetic Voice Detection

Sistema de detección de llamadas sintéticas o automatizadas basado en el análisis de patrones temporales de conversación y características acústicas de la voz.

El proyecto busca distinguir entre llamadas realizadas por una persona y llamadas generadas o atendidas mediante sistemas sintéticos, utilizando principalmente señales derivadas de la dinámica de la conversación.

Descripción

Los sistemas de voz sintética modernos pueden producir audio con una calidad suficientemente alta como para que la detección basada únicamente en características acústicas sea difícil.

Este proyecto utiliza una estrategia de fusión de características temporales y acústicas.

La hipótesis principal es que una llamada sintética no solamente puede diferenciarse por cómo suena la voz, sino también por cómo responde dentro de una conversación.

Una arquitectura típica de voz sintética puede involucrar:

Audio del usuario
      |
      v
     ASR
      |
      v
    LLM
      |
      v
     TTS
      |
      v
Respuesta sintética

Cada etapa introduce procesamiento adicional. Esto puede generar patrones temporales diferentes a los de una conversación entre dos personas.

El detector explota principalmente estos patrones.

⸻

Arquitectura

El pipeline completo es:

                    WAV
                     |
                     v
              Feature Extraction
                (features.py)
                     |
          +----------+----------+
          |                     |
          v                     v
   Timing Features       Acoustic Features
       (30)                   (26)
          |                     |
          v                     v
   Logistic Regression   Logistic Regression
          |                     |
          +----------+----------+
                     |
                     v
               Late Fusion
                     |
                     v
             Calibration
                     |
                     v
        P(synthetic | audio)
                     |
                     v
             Final Decision

El modelo final trabaja con:

* 30 características temporales.
* 26 características acústicas.
* 57 características disponibles en total en el extractor, incluyendo el nivel de la señal del caller.
* Dos clasificadores independientes.
* Un modelo de late fusion.
* Calibración probabilística opcional.

⸻

Estructura del proyecto

.
├── app.py
├── baseline.py
├── detector.py
├── features.py
├── model.py
├── schema.py
├── manifest.csv
├── features_cache.csv
├── detector.joblib
└── baseline.joblib

features.py

Contiene el procesamiento de audio y extracción de características.

Responsabilidades principales:

* Leer archivos WAV.
* Separar los canales de caller y agent.
* Normalizar y remuestrear el audio.
* Detectar segmentos de voz mediante VAD.
* Calcular latencias entre participantes.
* Extraer características temporales.
* Extraer MFCCs.
* Construir el vector de características.

detector.py

Contiene los modelos estadísticos utilizados por el sistema.

Incluye:

* Clasificadores logísticos.
* LateFusionDetector.
* ThresholdBaseline.
* Estimación de umbrales mediante Neyman-Pearson.
* Modelo posterior para las latencias.
* Fusión de múltiples ramas.

model.py

Entrena el detector principal.

El flujo es:

manifest.csv
      |
      v
features.py
      |
      v
features_cache.csv
      |
      v
Train / Validation Split
      |
      v
Timing Model + Acoustic Model
      |
      v
Out-of-Fold Predictions
      |
      v
Meta Model
      |
      v
Calibration
      |
      v
detector.joblib

baseline.py

Entrena y evalúa un detector más sencillo basado exclusivamente en las latencias de respuesta.

Sirve como referencia para comparar el modelo completo contra una estrategia estadística más simple.

app.py

Expone el detector mediante una API REST utilizando FastAPI.

Recibe un archivo WAV, extrae sus características y devuelve la probabilidad estimada de que la llamada sea sintética.

schema.py

Define el esquema de respuesta de la API mediante Pydantic.

⸻

Extracción de características

Procesamiento del audio

El sistema espera un WAV estéreo:

Channel 0 -> Caller
Channel 1 -> Agent

El audio se remuestrea a:

8 kHz

El procesamiento utiliza ventanas de:

20 ms

A 8 kHz esto corresponde a:

0.020 × 8000 = 160 samples

Cada ventana se analiza para determinar si contiene voz.

⸻

Voice Activity Detection

El VAD se basa en la energía de cada frame.

Para cada frame se calcula aproximadamente:

[
E = \frac{1}{N}\sum_{i=1}^{N}x_i^2
]

y posteriormente se expresa en decibeles:

[
E_{dB}=10\log_{10}(E)
]

El sistema estima un piso de ruido utilizando un percentil bajo de las energías y establece un umbral adaptativo.

Esto permite separar:

Audio
 |
 +-- Speech
 |
 +-- Silence / Noise

Los segmentos consecutivos de voz se agrupan y se aplican reglas de:

* Padding.
* Merging de segmentos cercanos.
* Duración mínima.

⸻

Latencias conversacionales

Una de las características principales del proyecto es la latencia de respuesta.

Para cada respuesta del agent se busca la siguiente intervención del caller.

La latencia se define como:

[
\lambda =
t_{\text{start,c​​aller}}

t_{\text{end,agent}}
]

Por ejemplo:

Agent:  |==========|
Caller:             |========|
                   ^
                   |
                latency

Si:

[
\lambda > 0
]

existe silencio entre las intervenciones.

Si:

[
\lambda < 0
]

existe solapamiento o barge-in.

Agent:  |==========|
Caller:        |==========|
                    ^
                    |
               lambda < 0

Esto permite capturar diferencias en la dinámica conversacional.

⸻

Características temporales

El extractor calcula características relacionadas con:

Actividad de voz

* Proporción de tiempo hablado.
* Número de segmentos.
* Duración promedio.
* Duración mediana.
* Variabilidad de duración.

Latencias

Se calculan estadísticas como:

* Media.
* Mediana.
* Desviación estándar.
* Percentiles.
* Proporción de respuestas lentas.
* Proporción de respuestas rápidas.
* Proporción de solapamientos.
* Skewness.
* Kurtosis.
* Coeficiente de variación.

También se calcula la correlación entre:

[
\text{latencia}
]

y

[
\text{duración de la respuesta}
]

Estas características permiten representar la conversación como una estructura temporal y no simplemente como una secuencia de muestras de audio.

⸻

Características acústicas

La rama acústica utiliza MFCCs (Mel-Frequency Cepstral Coefficients).

Los MFCCs representan características espectrales del audio relacionadas con la percepción humana de frecuencias.

El pipeline conceptual es:

Waveform
   |
   v
Framing
   |
   v
FFT
   |
   v
Mel Filter Bank
   |
   v
Log Energies
   |
   v
DCT
   |
   v
MFCCs

Para cada llamada se calculan:

* 13 medias de MFCC.
* 13 desviaciones estándar de MFCC.

Por lo tanto:

[
13 + 13 = 26
]

características acústicas.

Los MFCCs se calculan sobre regiones identificadas como voz, evitando que el silencio domine la representación acústica.

⸻

Representación matemática

Cada llamada puede representarse como un punto en un espacio de características:

[
x_i \in \mathbb{R}^{57}
]

Por lo tanto, un dataset con (n) llamadas puede representarse como:

[
X \in \mathbb{R}^{n\times57}
]

Sin embargo, el modelo principal separa las características en dos bloques:

[
x_i =
\begin{bmatrix}
x_i^{timing}\
x_i^{acoustic}
\end{bmatrix}
]

donde:

[
x_i^{timing}\in\mathbb{R}^{30}
]

y

[
x_i^{acoustic}\in\mathbb{R}^{26}
]

El nivel original del caller se conserva como una característica adicional para análisis y control de posibles confusores.

⸻

Modelo

Logistic Regression

Cada rama utiliza un pipeline:

StandardScaler
      |
      v
LogisticRegression

La regresión logística calcula:

[
P(y=1|x)=
\sigma(w^Tx+b)
]

donde:

[
\sigma(z)=\frac{1}{1+e^{-z}}
]

El hiperplano:

[
w^Tx+b=0
]

separa las regiones del espacio de características asociadas con las dos clases.

En este proyecto:

y = 0 -> Human
y = 1 -> Synthetic

⸻

Late Fusion

En lugar de combinar todas las características directamente en un solo clasificador, se entrenan dos modelos independientes:

Timing Features
      |
      v
Timing Model
      |
      v
P(synthetic | timing)

y:

Acoustic Features
      |
      v
Acoustic Model
      |
      v
P(synthetic | acoustic)

Después se combinan sus probabilidades:

[
p_t=P(S|x_t)
]

[
p_a=P(S|x_a)
]

y el meta-modelo aprende:

[
P(S|p_t,p_a)
]

Esto produce una arquitectura:

[
\mathbb{R}^{30}
\rightarrow
[0,1]
]

[
\mathbb{R}^{26}
\rightarrow
[0,1]
]

[
\mathbb{R}^{2}
\rightarrow
[0,1]
]

La ventaja es que la información temporal y acústica conserva su propia representación antes de fusionarse.

⸻

Out-of-Fold Predictions

Para entrenar el meta-modelo se utilizan predicciones out-of-fold.

El objetivo es evitar que el meta-modelo reciba predicciones generadas sobre los mismos datos utilizados para entrenar los modelos base.

El proceso es:

Training Set
     |
     +---- Fold 1
     +---- Fold 2
     +---- Fold 3
     +---- Fold 4
     +---- Fold 5

Cada modelo base predice los ejemplos que no utilizó durante su entrenamiento.

Estas predicciones se utilizan posteriormente para entrenar el modelo de fusión.

Esto reduce el riesgo de data leakage y de que el meta-modelo aprenda a partir de predicciones artificialmente optimistas.

⸻

Calibración

El modelo no solamente busca clasificar:

Human / Synthetic

sino producir una probabilidad:

0.03
0.27
0.71
0.94

Por eso se incluye una etapa opcional de calibración.

La calibración busca que valores como:

[
P(S)=0.8
]

correspondan aproximadamente a una frecuencia real de 80% de casos sintéticos dentro de ejemplos con esa probabilidad.

Se soportan métodos como:

* Sigmoid calibration.
* Isotonic calibration.

⸻

Baseline estadístico

Además del modelo de machine learning existe un baseline basado únicamente en las latencias.

Se define una variable binaria:

[
Z_i =
\begin{cases}
1 & \lambda_i > \tau\
0 & \lambda_i \leq \tau
\end{cases}
]

donde actualmente:

[
\tau=2\text{ segundos}
]

Para una llamada se calcula:

[
q=
\frac{#{\lambda_i>\tau}}{r}
]

donde (r) es el número de latencias observadas.

La intuición es sencilla:

Si una llamada contiene una proporción inusualmente alta de respuestas con más de dos segundos de latencia, aumenta la evidencia de que la llamada es sintética.

⸻

Neyman-Pearson

El baseline estima:

[
p_H=P(\lambda>\tau|Human)
]

y:

[
p_S=P(\lambda>\tau|Synthetic)
]

Con estas probabilidades se deriva un umbral de decisión utilizando el principio de Neyman-Pearson.

La razón de verosimilitud para una observación binaria es:

[
\Lambda(k,r)=
\frac{
p_S^k(1-p_S)^{r-k}
}{
p_H^k(1-p_H)^{r-k}
}
]

donde (k) representa el número de respuestas lentas.

Esto permite transformar la frecuencia observada de respuestas lentas en evidencia estadística a favor de la hipótesis de llamada sintética.

⸻

Multi-Branch Fusion

El proyecto también contiene una implementación generalizada para fusionar múltiples ramas probabilísticas.

Se soportan diferentes estrategias:

Mean

Promedio de las probabilidades:

[
p=
\frac{1}{K}\sum_{i=1}^{K}p_i
]

Stacking

Las probabilidades de cada rama se utilizan como variables de entrada de una regresión logística.

Log-Odds Fusion

Cada probabilidad se transforma a log-odds:

[
\operatorname{logit}(p)

\log\left(\frac{p}{1-p}\right)
]

y posteriormente se aprende una combinación lineal de estos valores.

Esta representación es especialmente útil cuando se quiere interpretar la evidencia aportada por cada rama.

⸻

Entrenamiento

El archivo manifest.csv define los datos utilizados durante el entrenamiento.

Conceptualmente contiene información similar a:

path,label,split
data/call_001.wav,0,train
data/call_002.wav,1,train
data/call_003.wav,0,val
data/call_004.wav,1,val

donde:

0 = Human
1 = Synthetic

Durante el entrenamiento:

1. Se lee el manifest.
2. Se procesa cada WAV.
3. Se extraen las características.
4. Las características se almacenan en cache.
5. Se divide el dataset en entrenamiento y validación.
6. Se entrenan los modelos de timing y acústica.
7. Se generan predicciones out-of-fold.
8. Se entrena el modelo de late fusion.
9. Se calibra la salida.
10. Se evalúa sobre validation.
11. Se guarda el modelo en detector.joblib.

El cache de características evita tener que procesar nuevamente todos los archivos de audio en cada ejecución.

⸻

Métricas

El entrenamiento calcula métricas como:

Accuracy

[
Accuracy=
\frac{TP+TN}{TP+TN+FP+FN}
]

Balanced Accuracy

Especialmente útil cuando las clases no están perfectamente balanceadas:

[
BalancedAccuracy=
\frac{TPR+TNR}{2}
]

ROC AUC

Mide la capacidad del modelo para ordenar correctamente ejemplos sintéticos por encima de ejemplos humanos independientemente de un umbral específico.

⸻

API

La aplicación utiliza FastAPI.

Ejecutar el servidor

Una vez entrenado el modelo:

uvicorn app:app --host 0.0.0.0 --port 8000

El modelo se carga al iniciar la aplicación desde:

detector.joblib

⸻

Endpoint

POST /detect

Recibe un archivo WAV o audio codificado en Base64.

El pipeline de inferencia es:

WAV
 |
 v
Feature Extraction
 |
 v
57 Features
 |
 v
DataFrame
 |
 v
Detector
 |
 v
Probability
 |
 v
Threshold
 |
 v
JSON

Respuesta:

{
  "is_synthetic": true,
  "confidence": 0.91
}

La confianza se calcula como:

[
\max(P(S),1-P(S))
]

⸻

GET /health

Permite comprobar que la API está funcionando y si el detector fue cargado correctamente.

Ejemplo:

{
  "status": "ok",
  "model_loaded": true
}

⸻

Umbral de decisión

Por defecto, la decisión utiliza:

[
threshold=0.5
]

Por lo tanto:

[
P(S)\geq0.5
\Rightarrow Synthetic
]

y:

[
P(S)<0.5
\Rightarrow Human
]

El umbral puede modificarse mediante la variable de entorno:

DECISION_THRESHOLD=0.7

Esto permite cambiar el comportamiento del sistema dependiendo de si se quiere priorizar:

* Reducir falsos positivos.
* Reducir falsos negativos.
* Aumentar sensibilidad.
* Aumentar especificidad.

⸻

Instalación

Instalar las dependencias necesarias:

pip install numpy pandas scipy scikit-learn librosa soundfile fastapi uvicorn joblib pydantic

También puede utilizarse un entorno virtual:

python -m venv .venv
source .venv/bin/activate

En Windows:

.venv\Scripts\activate

⸻

Flujo de uso

1. Preparar el dataset

Organizar los archivos WAV y crear un manifest.csv.

data/
├── human/
│   ├── call_001.wav
│   └── call_002.wav
└── synthetic/
    ├── call_003.wav
    └── call_004.wav

2. Entrenar el detector

python model.py

Esto genera:

detector.joblib
features_cache.csv

3. Entrenar el baseline

python baseline.py

Esto genera:

baseline.joblib

4. Iniciar la API

uvicorn app:app --host 0.0.0.0 --port 8000

5. Realizar una detección

Enviar un WAV al endpoint:

POST /detect

El sistema devuelve una probabilidad y una decisión binaria.

⸻

Limitaciones

El sistema depende de varias suposiciones sobre el audio de entrada.

Formato

El extractor espera un audio estéreo donde:

Channel 0 = Caller
Channel 1 = Agent

Un archivo mono o con canales intercambiados puede producir resultados incorrectos.

VAD

La extracción de segmentos depende de un detector de actividad de voz basado en energía. Ruido, música, compresión o grabaciones de baja calidad pueden afectar la segmentación.

Latencia

La latencia es una señal útil, pero no es exclusiva de sistemas sintéticos.

Una persona puede responder lentamente debido a:

* Problemas de red.
* Distracciones.
* Mala calidad de audio.
* Procesamiento cognitivo.
* Transferencias.
* Pausas naturales.

De forma similar, un sistema sintético suficientemente optimizado puede responder con baja latencia.

Generalización

El desempeño puede cambiar cuando el sistema encuentra:

* Nuevos modelos TTS.
* Nuevos modelos ASR.
* Nuevos LLMs.
* Diferentes codecs.
* Diferentes condiciones de red.
* Nuevos acentos o idiomas.
* Diferentes micrófonos.
* Diferentes tipos de conversación.

Por esta razón, las métricas obtenidas sobre un dataset específico no deben interpretarse automáticamente como desempeño universal.

⸻

Idea central

La principal idea del proyecto es que detectar una voz sintética no necesariamente requiere analizar únicamente su contenido acústico.

Una conversación puede representarse como una estructura temporal:

Caller      Agent       Caller       Agent
 |-----------|           |------------|
             <--- λ --->

y las características de esas interacciones pueden contener información sobre el sistema que produjo la respuesta.

El detector combina dos fuentes de evidencia:

[
\boxed{
\text{Synthetic Detection}

\text{Temporal Evidence}
+
\text{Acoustic Evidence}
}
]

La rama temporal busca identificar patrones producidos por la interacción con sistemas automatizados, mientras que la rama acústica busca capturar características de la señal de voz.

La combinación de ambas fuentes permite construir un detector que no depende de una sola característica de la voz, sino de la estructura completa de la interacción.

