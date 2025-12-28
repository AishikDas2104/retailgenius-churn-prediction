import pandas as pd
from sklearn.impute import SimpleImputer

RAW_DATA_PATH = "data/raw/E_Commerce_Dataset.xlsx"
PROCESSED_DATA_PATH = "data/processed/clean_data.csv"


def preprocess_data():
    # Load dataset
    df = pd.read_excel(RAW_DATA_PATH, sheet_name="E Comm")

    # Drop CustomerID (identifier, not useful for ML)
    df = df.drop(columns=["CustomerID"])

    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    # Impute missing values
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    # Save processed data
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print("Data preprocessing completed. Saved to:", PROCESSED_DATA_PATH)


if __name__ == "__main__":
    preprocess_data()
