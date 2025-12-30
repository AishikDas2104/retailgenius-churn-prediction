import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

DATA_PATH = "data/processed/features.csv"
MODEL_PATH = "models/churn_model.pkl"


def run_shap():
    # Load data
    df = pd.read_csv(DATA_PATH)
    X = df.drop("Churn", axis=1)

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Create SHAP explainer
    explainer = shap.Explainer(model, X)

    # Compute SHAP values
    shap_values = explainer(X)

    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("shap_summary_plot.png", bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.savefig("shap_mean_plot.png", bbox_inches="tight")
    plt.close()

    local_explanation = shap_values[0, :, 1]

    shap.plots.waterfall(local_explanation, show=False)
    plt.savefig("shap_waterfall.png", bbox_inches="tight")
    plt.close()

    shap.plots.force(
        local_explanation,
        matplotlib=True,
        show=False
    )
    plt.savefig("shap_force.png", bbox_inches="tight")
    plt.close()
    

    print(" SHAP plots generated successfully.")


if __name__ == "__main__":
    run_shap()
