# Taller 4 — MLflow + GitHub Actions
# Makefile: Comandos para gestionar el pipeline de ML
#
# Uso:
#   make install    → Instala las dependencias de Python
#   make train      → Entrena el modelo y lo registra en MLflow
#   make validate   → Valida el modelo cargándolo desde el registro de MLflow
#   make all        → Ejecuta install → train → validate en secuencia

## Instala las dependencias definidas en requirements.txt
install:
	pip install -r requirements.txt

## Entrena el modelo con el dataset Wine Quality y lo registra en MLflow (genera mlruns/)
train:
	python train.py

## Valida el modelo cargándolo desde el registro local de MLflow
## Retorna error si el modelo no cumple los umbrales de calidad (MSE, R²)
validate:
	python validate.py

## Pipeline completo: instalar dependencias → entrenar → validar
all: install train validate

.PHONY: install train validate all
