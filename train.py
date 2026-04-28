"""
Taller 4 — MLflow + GitHub Actions
train.py: Entrenamiento y registro del modelo con MLflow

Dataset: Diabetes Dataset (Efron et al., 2004)
  Archivo: diabetes.csv (incluido en el repositorio)
  Descripción: 442 pacientes con diabetes. Contiene 10 variables clínicas
               normalizadas (edad, sexo, IMC, presión arterial y 6 marcadores
               sanguíneos) y una medida cuantitativa de progresión de la
               enfermedad al año siguiente como variable objetivo.
  Fuente original: https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
  Tarea: Regresión — predecir disease_progression (progresión de la enfermedad)

Modelo: Random Forest Regressor
Métricas: MSE (error cuadrático medio), R² (coeficiente de determinación)
"""

import os
import sys
import traceback

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── Configuración ──────────────────────────────────────────────────────────────
DATASET_FILE = "diabetes.csv"
TARGET_COL = "disease_progression"
EXPERIMENT_NAME = "diabetes-progression-experiment"
MODEL_NAME = "diabetes-rf"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Hiperparámetros del modelo
N_ESTIMATORS = 150
MAX_DEPTH = 8
MIN_SAMPLES_SPLIT = 4

# ── Rutas MLflow ───────────────────────────────────────────────────────────────
workspace_dir = os.getcwd()
mlruns_dir = os.path.join(workspace_dir, "mlruns")
tracking_uri = "file://" + os.path.abspath(mlruns_dir)

os.makedirs(mlruns_dir, exist_ok=True)
mlflow.set_tracking_uri(tracking_uri)

print(f"[INFO] CWD: {workspace_dir}")
print(f"[INFO] Tracking URI: {tracking_uri}")

# ── Cargar dataset desde CSV ───────────────────────────────────────────────────
data_path = os.path.join(workspace_dir, DATASET_FILE)
if not os.path.exists(data_path):
    print(f"[ERROR] No se encontró el archivo '{DATASET_FILE}' en {workspace_dir}")
    print("[ERROR] Asegúrate de que el CSV esté en la raíz del proyecto.")
    sys.exit(1)

df = pd.read_csv(data_path)
print(f"[INFO] Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"[INFO] Columnas: {list(df.columns)}")
print(f"[INFO] Target '{TARGET_COL}' — rango: [{df[TARGET_COL].min():.1f}, {df[TARGET_COL].max():.1f}]")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"[INFO] Entrenamiento: {X_train.shape} | Test: {X_test.shape}")

# ── Crear / recuperar experimento MLflow ───────────────────────────────────────
# set_experiment crea el experimento si no existe, o lo recupera si ya existe
experiment = mlflow.set_experiment(EXPERIMENT_NAME)
experiment_id = experiment.experiment_id
print(f"[INFO] Experimento '{EXPERIMENT_NAME}' activo (ID: {experiment_id})")

# ── Entrenar y registrar modelo en MLflow ─────────────────────────────────────
try:
    with mlflow.start_run(experiment_id=experiment_id) as run:
        print(f"[INFO] Run iniciado: {run.info.run_id}")

        # Entrenar modelo
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Calcular métricas
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"[INFO] MSE: {mse:.4f} | R²: {r2:.4f}")

        # Registrar parámetros
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("max_depth", MAX_DEPTH)
        mlflow.log_param("min_samples_split", MIN_SAMPLES_SPLIT)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("dataset", "diabetes.csv (Efron et al., 2004)")
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", X_train.shape[0])

        # Registrar métricas
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        # Firma e input_example (criterio "Se excede" de la rúbrica)
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.head(5)

        # Registrar modelo en el Model Registry de MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=MODEL_NAME,
        )

        print(f"[INFO] Artefactos en: {run.info.artifact_uri}")
        print(f"\n✅ Modelo '{MODEL_NAME}' registrado. MSE={mse:.4f} | R²={r2:.4f}")

except Exception:
    print("[ERROR] Fallo durante el entrenamiento o registro del modelo:")
    traceback.print_exc()
    sys.exit(1)
