import pandas as pd
from sklearn.preprocessing import LabelEncoder

INPUT_PATH = "data/processed/clean_data.csv"
OUTPUT_PATH = "data/processed/features.csv"


def engineer_features():
    df = pd.read_csv(INPUT_PATH)

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    df.to_csv(OUTPUT_PATH, index=False)
    print("Feature engineering completed. Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    engineer_features()
