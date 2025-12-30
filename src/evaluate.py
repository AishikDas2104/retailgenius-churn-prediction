import pandas as pd
import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/features.csv"
MODEL_PATH = "models/churn_model.pkl"


def evaluate_model():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(X_test)

    print("Model Evaluation Report:")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    evaluate_model()
