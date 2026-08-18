#archivo que contiene el preprocesamiento
#1. Corrige los valores de edad 0 y códigos 96/68 en las variables de atraso
#2. Imputa los valores faltantes con la Mediana del conjunto de entrenamiento

#importación de librerías
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer

#variables de atraso donde aparecen los códigos anómalos 96 y 98
LATE_PAYMENT_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]


#función que reemplaza los valores inválidos por NaN para despues imputarlos
def fix_structural_errors(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    if "age" in X.columns:
        X.loc[X["age"] == 0, "age"] = np.nan

    for col in LATE_PAYMENT_COLS:
        if col in X.columns:
            X.loc[X[col] >= 96, col] = np.nan

    return X


#función que crea un pipeline que corrige errores y completa valores faltantes con la mediana
def build_preprocessor() -> Pipeline:
    #convertir la función fix_structural_errors compatible con sklearn
    corrector = FunctionTransformer(
        fix_structural_errors,
        feature_names_out="one-to-one",
        validate=False,
    )

    preprocessor = Pipeline(steps=[
        ("fix_errors", corrector),
        ("imputer", SimpleImputer(strategy="median")),
    ])

    return preprocessor

#imputamos con la mediana en vez de la media ya que con unos pocos clientes con ingresos gigantes
#arrastrarían la media hacia arriba y estaríamos subestimando el riesgo. 