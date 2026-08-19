from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.features.preprocessing import build_preprocessor


RANDOM_STATE = 42

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