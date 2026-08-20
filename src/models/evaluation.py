#métricas de evaluación para el riesgo crediticio
#ROC-AUC, KS, GINI)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

#transformación directa del AUC
def gini_from_auc(auc:float) -> float:
    return 2 * auc - 1

#Kolmogorov-Smirnov mide que tanto el modelo sepra buenos de malos pagadores
#máxima sepración de las distribuciones acumuladas de verdaderos positivos y falsos positivos
def ks_statistic(y_true, y_proba) -> float:
    frp, tpr, _ = roc_curve(y_true, y_proba)
    return np.max(tpr - frp)

#devuleve un diccionario con las métricas clave de riesgo
def risk_metrics(y_true, y_proba) -> dict:
    auc = roc_auc_score(y_true, y_proba)
    return{
        "ROC_AUC": round(auc,4),
        "Gini": round(gini_from_auc(auc),4),
        "KS": round(ks_statistic(y_true, y_proba),4),
    }

#extraer los coeficients de las regresión logística
def coefficient_importance(pipeline, feature_names) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    coefs = model.coef_[0]
    df = pd.DataFrame({
        "variable": feature_names,
        "coeficiente": coefs,
        "odds_ratio": np.exp(coefs),
    })
    df["abs_coef"] = df["coeficiente"].abs()
    df = df.sort_values("abs_coef", ascending=False).drop(columns="abs_coef")
    return df.reset_index(drop=True)