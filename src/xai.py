import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
from typing import Any, Optional

def generate_shap_explanations(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    output_dir: str = "outputs"
) -> shap.Explainer:
    """
    Generates SHAP Global Feature Importance and Summary Plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n[XAI - SHAP] Calculating SHAP values for global interpretability...")

    # For VotingClassifier or generic pipelines, use KernelExplainer or Sampling
    # Using background summary sample for computational efficiency
    background = shap.kmeans(X_train, min(25, len(X_train)))
    
    # Define prediction function for probabilities
    predict_fn = lambda x: model.predict_proba(x)[:, 1] if hasattr(model, 'predict_proba') else model.predict(x)
    explainer = shap.KernelExplainer(predict_fn, background)
    
    sample_test = X_test.iloc[:min(30, len(X_test))]
    shap_values = explainer.shap_values(sample_test)

    # 1. Summary Bar Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample_test, plot_type="bar", show=False)
    plt.title("SHAP Global Feature Importance (PCOS Prediction)")
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "shap_feature_importance.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()

    # 2. Beeswarm Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample_test, show=False)
    plt.title("SHAP Summary Beeswarm Plot")
    plt.tight_layout()
    beeswarm_path = os.path.join(output_dir, "shap_beeswarm_plot.png")
    plt.savefig(beeswarm_path, dpi=300)
    plt.close()

    print(f"SHAP plots saved to: {bar_path} and {beeswarm_path}")
    return explainer

def generate_lime_explanation(
    model: Any,
    X_train: pd.DataFrame,
    instance: pd.Series,
    instance_idx: int = 0,
    class_names: list = ["Non-PCOS", "PCOS"],
    output_dir: str = "outputs"
) -> lime.lime_tabular.LimeTabularExplainer:
    """
    Generates LIME Local Explanation for an individual patient instance.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[XAI - LIME] Generating local explanation for instance #{instance_idx}...")

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=list(X_train.columns),
        class_names=class_names,
        mode="classification",
        random_state=42
    )

    exp = explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict_proba,
        num_features=min(10, X_train.shape[1])
    )

    html_path = os.path.join(output_dir, f"lime_explanation_patient_{instance_idx}.html")
    exp.save_to_file(html_path)

    fig = exp.as_pyplot_figure()
    plt.title(f"LIME Local Explanation - Patient #{instance_idx}")
    plt.tight_layout()
    png_path = os.path.join(output_dir, f"lime_explanation_patient_{instance_idx}.png")
    plt.savefig(png_path, dpi=300)
    plt.close()

    print(f"LIME explanation saved to: {html_path} and {png_path}")
    return explainer
