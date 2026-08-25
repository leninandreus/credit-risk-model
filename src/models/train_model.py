from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.features.preprocessing import build_preprocessor


RANDOM_STATE = 42
SCALE_POS_WEIGHT = 14

def buil_logistic_pipeline() -> Pipeline:
    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("scaler", StandardScaler()),
        ("model",LogisticRegression(
            class_weight = "balanced",
            max_iter = 1000,
            random_state = RANDOM_STATE
        )),

    ])
    return pipeline

def build_random_forest_pipeline(n_estimators = 300) -> Pipeline:
    return Pipeline(steps=[
        ("preprocessor",build_preprocessor()),
        ("model",RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])

def build_xgboost_pipeline() -> Pipeline:
    return Pipeline(steps=[
        ("preprocessor",build_preprocessor()),
        ("model",XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=SCALE_POS_WEIGHT, #compensa el desbalance
            eval_metric="auc",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            )),
        ])