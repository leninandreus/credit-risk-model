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

#evalúa un pipeline con validación cruzada sobre el train para comparar todos los modelos
def evaluate_model_cv(pipeline,X_train,y_train,cv_splits=5, random_state=42):
    from sklearn.model_selection import cross_val_score,StratifiedKFold

    cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipeline, X_train, y_train,cv=cv, scoring="roc_auc", n_jobs=-1)

    return{
        "roc_auc_mean": round(scores.mean(),4),
        "roc_auc_std": round(scores.std(),4),
        "scores":np.round(scores,4),
    }

#evalúa sobre el train completo y sobre el test, devuelve métricas de riesgo, probabilidades y predicciones
def evaluate_on_test(pipeline,X_train,y_train,X_test,y_test, threshold=0.5):
    pipeline.fit(X_train,y_train)
    y_proba = pipeline.predict_proba(X_test)[:,1]
    y_pred = (y_proba >= threshold).astype(int)
    return risk_metrics(y_test,y_proba),y_proba,y_pred


