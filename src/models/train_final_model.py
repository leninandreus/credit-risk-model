#entrenamiento del modelo ganador (XGBoost) sobre el conjunto de entrenamiento
#se guarda un joblib listo para producción

#vamos a guardar el pipeline completo (preprocesamiento + modelo) para que cuando lleguen datos nuevos
#basta con cargar este archivo y pasarle datos crudos.
from pathlib import Path
import joblib

from src.data.load_split import load_raw_data, split_data
from src.models.train_model import build_xgboost_pipeline

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH = MODEL_DIR / "xgboost_final.joblib"

def train_and_save():
    print("-----Cargando datos y haciendo la partición---------")
    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df)

    print("---------Entrenando el pipeline XGBoost sobre el conjunto train completo------")
    pipeline = build_xgboost_pipeline()
    pipeline.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"----Modelo guardado en : {MODEL_PATH}")
    return pipeline

if __name__=="__main__":
    train_and_save()


