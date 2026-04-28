"""
Taller 4 — MLflow + GitHub Actions
validate.py: Validación del modelo registrado en MLflow

Carga el modelo más reciente desde el registro local de MLflow
y lo evalúa con datos de validación externos (diabetes.csv).

El script retorna exit code 0 si el modelo cumple los criterios de calidad,
o exit code 1 si falla — lo que detiene el pipeline de GitHub Actions.
"""

import os
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── Configuración ──────────────────────────────────────────────────────────────
DATASET_FILE = "diabetes.csv"
TARGET_COL = "disease_progression"
MODEL_NAME = "diabetes-rf"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Umbrales de calidad aceptable
MSE_THRESHOLD = 3500.0  # MSE máximo permitido
R2_MIN = 0.40           # R² mínimo aceptable

# ── Rutas MLflow ───────────────────────────────────────────────────────────────
workspace_dir = os.getcwd()
mlruns_dir = os.path.join(workspace_dir, "mlruns")
tracking_uri = "file://" + os.path.abspath(mlruns_dir)
mlflow.set_tracking_uri(tracking_uri)

print(f"[INFO] CWD: {workspace_dir}")
print(f"[INFO] Tracking URI: {tracking_uri}")

# ── Cargar datos de validación ─────────────────────────────────────────────────
data_path = os.path.join(workspace_dir, DATASET_FILE)
if not os.path.exists(data_path):
    print(f"[ERROR] No se encontró '{DATASET_FILE}' en {workspace_dir}")
    sys.exit(1)

df = pd.read_csv(data_path)
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Usar el mismo split que train.py (misma semilla) → evaluar sobre datos no vistos
_, X_val, _, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"[INFO] Datos de validación: {X_val.shape[0]} muestras x {X_val.shape[1]} features")

# ── Cargar modelo desde el registro de MLflow ──────────────────────────────────
model_uri = f"models:/{MODEL_NAME}/latest"
print(f"[INFO] Cargando modelo desde MLflow Registry: {model_uri}")

try:
    model = mlflow.sklearn.load_model(model_uri)
    print("[INFO] Modelo cargado correctamente")
except Exception as e:
    print(f"[ERROR] No se pudo cargar el modelo '{model_uri}': {e}")
    print("[ERROR] Asegúrate de que 'make train' se ejecutó antes de 'make validate'.")
    sys.exit(1)

# ── Predicción y evaluación ────────────────────────────────────────────────────
preds = model.predict(X_val)
mse = mean_squared_error(y_val, preds)
r2 = r2_score(y_val, preds)

print(f"\n{'='*50}")
print(f"  Resultados de validación — {MODEL_NAME}")
print(f"{'='*50}")
print(f"  MSE : {mse:.4f}  (umbral ≤ {MSE_THRESHOLD})")
print(f"  R²  : {r2:.4f}  (mínimo ≥ {R2_MIN})")
print(f"{'='*50}\n")

# ── Criterios de calidad ───────────────────────────────────────────────────────
mse_ok = mse <= MSE_THRESHOLD
r2_ok = r2 >= R2_MIN

if mse_ok and r2_ok:
    print(
        f"✅ El modelo cumple los criterios de calidad. "
        f"MSE={mse:.4f} ≤ {MSE_THRESHOLD} | R²={r2:.4f} ≥ {R2_MIN}"
    )
    sys.exit(0)
else:
    if not mse_ok:
        print(f"❌ MSE fuera de rango: {mse:.4f} > {MSE_THRESHOLD}")
    if not r2_ok:
        print(f"❌ R² insuficiente: {r2:.4f} < {R2_MIN}")
    print("Pipeline detenido — el modelo no cumple los estándares de calidad.")
    sys.exit(1)
