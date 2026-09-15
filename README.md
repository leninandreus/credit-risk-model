````markdown
# Modelo de Riesgo Crediticio

Proyecto de Machine Learning para estimar la probabilidad de que un cliente presente una mora grave durante los próximos dos años.

El objetivo es construir un modelo de *credit scoring* que permita clasificar clientes según su nivel de riesgo y apoyar decisiones de otorgamiento de crédito.

**Dataset:** [Give Me Some Credit - Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)

---

## Metodología

El proyecto incluye:

- Análisis exploratorio de datos.
- Partición train/test estratificada.
- Preprocesamiento mediante pipelines.
- Tratamiento de valores faltantes.
- Comparación de distintos algoritmos.
- Validación cruzada.
- Evaluación mediante ROC-AUC, Gini y KS.
- Interpretación del modelo final con SHAP.

Se compararon tres modelos:

- Regresión Logística
- Random Forest
- XGBoost

---

## Resultados

Resultados obtenidos sobre el conjunto de test:

| Modelo | ROC-AUC | Gini | KS |
|---|---:|---:|---:|
| **XGBoost** | **0.860** | **0.720** | **0.564** |
| Random Forest | 0.846 | 0.691 | 0.545 |
| Regresión Logística | 0.821 | 0.641 | 0.499 |

**XGBoost** obtuvo el mejor desempeño en las tres métricas y fue seleccionado como modelo final.

El análisis con SHAP mostró que variables relacionadas con la utilización del crédito y el historial de atrasos tienen una influencia importante sobre la probabilidad de incumplimiento.

---

## Estructura del proyecto

```text
credit-risk-model/
│
├── data/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_train_test_split.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_logistic_regression.ipynb
│   ├── 05_random_forest.ipynb
│   ├── 06_xgboost.ipynb
│   ├── 07_model_comparison.ipynb
│   └── 08_shap_interpretation.ipynb
│
├── src/
│   ├── data/
│   │   └── load_split.py
│   ├── features/
│   │   └── preprocessing.py
│   └── models/
│       ├── train_model.py
│       ├── evaluation.py
│       └── train_final_model.py
│
├── models/
├── requirements.txt
└── README.md
````

---

## Ejecución

Clonar el repositorio:

```bash
git clone https://github.com/leninandreus/credit-risk-model.git
cd credit-risk-model
```

Crear y activar el entorno virtual:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Descargar el dataset desde [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit) y guardarlo como:

```text
data/cs-training.csv
```

Para entrenar el modelo final:

```bash
python -m src.models.train_final_model
```

El pipeline completo de preprocesamiento y XGBoost se guarda en:

```text
models/xboost_final.joblib
```

---

## Tecnologías utilizadas

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `SHAP` · `matplotlib` · `seaborn` · `joblib`

---

## Autor

**Lenin Oñate**

[LinkedIn](https://www.linkedin.com/in/lenin11/)

```
```
