import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def create_base_paper_workflow():
    output_dir = r"c:\Users\user\OneDrive\Desktop\AI_ML\PCOS-Prediction-XAI\outputs"
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "base_paper_workflow.png")
    svg_path = os.path.join(output_dir, "base_paper_workflow.svg")

    fig = plt.figure(figsize=(19, 10), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    bg_color = "#F8FAFC"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    c_primary = "#1E293B"      # Slate 800
    c_blue = "#2563EB"         # Blue 600
    c_purple = "#7C3AED"       # Purple 600
    c_teal = "#0D9488"         # Teal 600
    c_green = "#16A34A"        # Green 600
    c_pink = "#DB2777"         # Pink 600
    c_card_bg = "#FFFFFF"
    c_border = "#CBD5E1"

    # Header Box
    title_box = FancyBboxPatch((2, 91), 96, 7.5, boxstyle="round,pad=0.5,rounding_size=1.2",
                               facecolor=c_primary, edgecolor="none", zorder=2)
    ax.add_patch(title_box)
    
    ax.text(50, 95.8, "BASELINE PAPER SYSTEM WORKFLOW (WILEY 2026)",
            color="#FFFFFF", fontsize=15, fontweight="bold", ha="center", va="center", fontfamily="sans-serif")
    ax.text(50, 93.0, "Two-Stage Feature Selection (Chi2 + CatBoost RFE) • Soft-Voting Ensemble (XGBoost + MLP) • SHAP & LIME",
            color="#94A3B8", fontsize=10.5, ha="center", va="center", fontfamily="sans-serif")

    # 5 Sequential Stages
    stages = [
        {
            "id": "STEP 1",
            "title": "Dataset &\nPreprocessing",
            "color": c_blue,
            "x": 2, "w": 18,
            "items": [
                "Kaggle Cohort (541 Patients, 44 Feats)",
                "Drop Sl No, Patient File No",
                "Mode Imputation for Missing Values",
                "MinMax Normalization into [0, 1]"
            ],
            "badge": "DATA PREPARATION"
        },
        {
            "id": "STEP 2",
            "title": "Leakage-Free Split &\nSMOTE Balancing",
            "color": c_purple,
            "x": 21.5, "w": 18,
            "items": [
                "80/20 Train-Test Stratified Split",
                "Train Set: 432 Samples (80%)",
                "SMOTE on Train Only (432 -> 582)",
                "Test Set Untouched (109 Samples)"
            ],
            "badge": "ZERO DATA LEAKAGE"
        },
        {
            "id": "STEP 3",
            "title": "Two-Stage Hybrid\nFeature Selection",
            "color": c_teal,
            "x": 41, "w": 18,
            "items": [
                "Stage 1: Chi-Square Filter (40 -> 30)",
                "Stage 2: CatBoost RFE (30 -> 16)",
                "Removes Collinear & Weak Feats",
                "Final 16 Critical Clinical Markers"
            ],
            "badge": "FILTER + WRAPPER"
        },
        {
            "id": "STEP 4",
            "title": "Machine Learning &\nSoft-Voting Ensemble",
            "color": c_green,
            "x": 60.5, "w": 18,
            "items": [
                "Evaluated 11 Base Classifiers",
                "Tree-Booster: XGBoost Classifier",
                "Neural Network: Multi-Layer Perceptron",
                "Soft-Voting: Average Class Probs"
            ],
            "badge": "CHAMPION: XGB + MLP"
        },
        {
            "id": "STEP 5",
            "title": "Evaluation &\nExplainable AI (XAI)",
            "color": c_pink,
            "x": 80, "w": 18,
            "items": [
                "Test Accuracy: 96.33% | F1: 96.30%",
                "10-Fold CV: 92.04% +/- 2.58%",
                "Global XAI: TreeSHAP Feature Rank",
                "Local XAI: LIME Single-Patient Plots"
            ],
            "badge": "96.33% ACCURACY"
        }
    ]

    for s in stages:
        x, w, col = s["x"], s["w"], s["color"]
        
        outer_box = FancyBboxPatch((x, 10), w, 78, boxstyle="round,pad=0.3,rounding_size=1.0",
                                   facecolor=c_card_bg, edgecolor=c_border, linewidth=1.5, zorder=2)
        ax.add_patch(outer_box)
        
        header_bar = FancyBboxPatch((x, 77), w, 11, boxstyle="round,pad=0.3,rounding_size=1.0",
                                    facecolor=col, edgecolor="none", zorder=3)
        ax.add_patch(header_bar)
        
        ax.text(x + w/2, 85.5, s["id"], color="#FFFFFF", fontsize=9, fontweight="bold",
                ha="center", va="center", fontfamily="sans-serif", alpha=0.9)
        ax.text(x + w/2, 81.0, s["title"], color="#FFFFFF", fontsize=11, fontweight="bold",
                ha="center", va="center", fontfamily="sans-serif", multialignment="center")
        
        badge_box = FancyBboxPatch((x + 1.5, 73), w - 3, 3.2, boxstyle="round,pad=0.2,rounding_size=0.6",
                                   facecolor=f"{col}15", edgecolor=col, linewidth=1.2, zorder=3)
        ax.add_patch(badge_box)
        ax.text(x + w/2, 74.6, s["badge"], color=col, fontsize=7.8, fontweight="bold",
                ha="center", va="center", fontfamily="sans-serif")
        
        for i, item in enumerate(s["items"]):
            y_pos = 57 - (i * 14.5)
            card = FancyBboxPatch((x + 1.2, y_pos), w - 2.4, 12, boxstyle="round,pad=0.3,rounding_size=0.8",
                                  facecolor="#F1F5F9", edgecolor="#E2E8F0", linewidth=1.0, zorder=3)
            ax.add_patch(card)
            
            circle = plt.Circle((x + 3.0, y_pos + 6.0), 1.6, color=col, zorder=4)
            ax.add_patch(circle)
            ax.text(x + 3.0, y_pos + 6.0, str(i + 1), color="#FFFFFF", fontsize=8.5, fontweight="bold",
                    ha="center", va="center", zorder=5)
            
            ax.text(x + 5.5, y_pos + 6.0, item, color=c_primary, fontsize=8.5, fontweight="medium",
                    ha="left", va="center", fontfamily="sans-serif", wrap=True, multialignment="left")

    for i in range(len(stages) - 1):
        x_start = stages[i]["x"] + stages[i]["w"]
        x_end = stages[i+1]["x"]
        y_mid = 48
        
        arrow = patches.FancyArrowPatch((x_start + 0.2, y_mid), (x_end - 0.2, y_mid),
                                        arrowstyle="simple,head_width=5,head_length=6",
                                        color=c_blue, linewidth=1.5, zorder=5)
        ax.add_patch(arrow)

    footer_box = FancyBboxPatch((2, 2.5), 96, 5.5, boxstyle="round,pad=0.4,rounding_size=0.8",
                                facecolor="#1E293B", edgecolor="none", zorder=2)
    ax.add_patch(footer_box)
    
    ax.text(50, 5.8, "Source: Health Science Reports (Wiley, 2026) • DOI: 10.1002/hsr2.73100",
            color="#38BDF8", fontsize=9.5, fontweight="bold", ha="center", va="center", fontfamily="sans-serif")
    ax.text(50, 3.8, "Title: A Two-Stage Hybrid Feature Selection and Ensemble Learning Framework With Explainable AI",
            color="#94A3B8", fontsize=8.5, ha="center", va="center", fontfamily="sans-serif")

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, facecolor=bg_color, bbox_inches="tight")
    plt.savefig(svg_path, format="svg", facecolor=bg_color, bbox_inches="tight")
    plt.close()
    print(f"Base Paper Diagram generated at:\n  PNG: {png_path}\n  SVG: {svg_path}")

if __name__ == "__main__":
    create_base_paper_workflow()
