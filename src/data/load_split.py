#archivo que realiza la carga de datos y la sepración de train y test
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


TARGET = "SeriousDlqin2yrs"
RANDOM_STATE = 42
TEST_SIZE = 0.20


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "cs-training.csv"

#función que carga el dataset crudo sin índice
def load_raw_data(path=None) -> pd.DataFrame:
    if path is None:
        path = DEFAULT_DATA_PATH

    df = pd.read_csv(path, index_col=0)
    return df

#función que divide el dataframe en train y test conteniendo una igual proporción de las clases de la variable objetivo
def split_data(df: pd.DataFrame,
               target: str = TARGET,
               test_size: float = TEST_SIZE,
               random_state: int = RANDOM_STATE):
 
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test