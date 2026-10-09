import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle

def create_publication_workflow():
    output_dir = r"c:\Users\user\OneDrive\Desktop\AI_ML\PCOS-Prediction-XAI\outputs"
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "research_methodology_workflow.png")
    svg_path = os.path.join(output_dir, "research_methodology_workflow.svg")

    # Canvas setup (High Resolution Publication Grade)
    fig = plt.figure(figsize=(20, 11), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color Palette (Nature/IEEE Medical Journals Standard)
    bg_color = "#F8FAFC"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    c_primary = "#0F172A"      # Slate 900
    c_blue = "#0284C7"         # Sky 600
    c_indigo = "#4F46E5"       # Indigo 600
    c_emerald = "#059669"      # Emerald 600
    c_rose = "#E11D48"         # Rose 600
    c_amber = "#D97706"        # Amber 600
    c_card_bg = "#FFFFFF"      # Pure White
    c_border = "#CBD5E1"       # Slate 300
    c_subtext = "#475569"      # Slate 600

    # Main Title Header Banner
    title_box = FancyBboxPatch((2, 91), 96, 7.5, boxstyle="round,pad=0.5,rounding_size=1.2",
                               facecolor=c_primary, edgecolor="none", zorder=2)
    ax.add_patch(title_box)
    
    ax.text(50, 95.8, "NEXT-GENERATION EXPLAINABLE AI & NEURO-TREE ENSEMBLE WORKFLOW FOR PCOS SCREENING",
            color="#FFFFFF", fontsize=15, fontweight="bold", ha="center", va="center", fontfamily="sans-serif")
    ax.text(50, 93.0, "Rotterdam Bio-Markers • Tri-Stage Feature Selection • Google TabNet & Boosters • DiCE Counterfactuals • Web CDSS",
            color="#94A3B8", fontsize=10.5, ha="center", va="center", fontfamily="sans-serif")

    # Define 5 Horizontal Stages
    stages = [
        {
            "id": "PHASE 1",
            "title": "Data Ingestion &\nLeakage-Free Prep",
            "color": c_blue,
            "x": 2, "w": 18,
            "items": [
                "Raw Clinical Cohort (541 Patients, 44 Feats)",
                "Outlier Audit & Missing Mode Imputation",
                "Strict Stratified 5x5 Nested CV Split",
                "Inside-Fold SMOTE-NC & RobustScaler"
            ],
            "badge": "ZERO LEAKAGE"
        },
        {
            "id": "PHASE 2",
            "title": "Rotterdam Biomarkers &\nTri-Stage Selection",
            "color": c_indigo,
            "x": 21.5, "w": 18,
            "items": [
                "22 Rotterdam Bio-Features (PCOM, LH/FSH)",
                "Stage 1: Chi2 & ANOVA Filter (44->30)",
                "Stage 2: Boruta + CatBoost RFE (30->18)",
                "Stage 3: ElasticNet Stability (Top 16)"
            ],
            "badge": "DIMENSION REDUCTION"
        },
        {
            "id": "PHASE 3",
            "title": "Calibrated Neuro-Tree\nSuper-Ensemble",
            "color": c_emerald,
            "x": 41, "w": 18,
            "items": [
                "12 Base Classifiers Benchmarked",
                "Bayesian Hyper-Tuning (Optuna Search)",
                "Stacking: TabNet + ResNet + XGB/CB/LGB",
                "Platt / Isotonic Probability Calibration"
            ],
            "badge": "CHAMPION MODEL (97.2% AUC)"
        },
        {
            "id": "PHASE 4",
            "title": "360° Clinical\nExplainability (XAI)",
            "color": c_rose,
            "x": 60.5, "w": 18,
            "items": [
                "Global TreeSHAP Beeswarm & Interactions",
                "Local LIME & Patient Waterfall Plots",
                "Novel DiCE Counterfactual Prescriptions",
                "Rotterdam Clinical Coherence Validation"
            ],
            "badge": "ACTIONABLE INSIGHTS"
        },
        {
            "id": "PHASE 5",
            "title": "Statistical Validation &\nClinical CDSS Web App",
            "color": c_amber,
            "x": 80, "w": 18,
            "items": [
                "DeLong AUC & Wilcoxon Tests (p < 0.001)",
                "Static Clinical Bedside Nomogram",
                "Interactive Streamlit Web Dashboard",
                "FastAPI Microservice & Model Registry"
            ],
            "badge": "BEDSIDE TRANSLATION"
        }
    ]

    # Draw Stage Containers & Cards
    for s in stages:
        x, w, col = s["x"], s["w"], s["color"]
        
        # Outer Stage Box
        outer_box = FancyBboxPatch((x, 10), w, 78, boxstyle="round,pad=0.3,rounding_size=1.0",
                                   facecolor=c_card_bg, edgecolor=c_border, linewidth=1.5, zorder=2)
        ax.add_patch(outer_box)
        
        # Top Header Accent Bar
        header_bar = FancyBboxPatch((x, 77), w, 11, boxstyle="round,pad=0.3,rounding_size=1.0",
                                    facecolor=col, edgecolor="none", zorder=3)
        ax.add_patch(header_bar)
        
        # Phase Tag
        ax.text(x + w/2, 85.5, s["id"], color="#FFFFFF", fontsize=9, fontweight="bold",
                ha="center", va="center", fontfamily="sans-serif", alpha=0.9)
        # Title
        ax.text(x + w/2, 81.0, s["title"], color="#FFFFFF", fontsize=11, fontweight="bold",
                ha="center", va="center", fontfamily="sans-serif", multialignment="center")
        
        # Badge
        badge_box = FancyBboxPatch((x + 1.5, 73), w - 3, 3.2, boxstyle="round,pad=0.2,rounding_size=0.6",
                                   facecolor=f"{col}15", edgecolor=col, linewidth=1.2, zorder=3)
        ax.add_patch(badge_box)
        ax.text(x + w/2, 74.6, s["badge"], color=col, fontsize=7.8, fontweight="bold",
                ha="center", va="center", fontfamily="sans-serif")
        
        # Inner Content Boxes (4 Step Cards per phase)
        for i, item in enumerate(s["items"]):
            y_pos = 57 - (i * 14.5)
            card = FancyBboxPatch((x + 1.2, y_pos), w - 2.4, 12, boxstyle="round,pad=0.3,rounding_size=0.8",
                                  facecolor="#F1F5F9", edgecolor="#E2E8F0", linewidth=1.0, zorder=3)
            ax.add_patch(card)
            
            # Step Number Circle
            circle = plt.Circle((x + 3.0, y_pos + 6.0), 1.6, color=col, zorder=4)
            ax.add_patch(circle)
            ax.text(x + 3.0, y_pos + 6.0, str(i + 1), color="#FFFFFF", fontsize=8.5, fontweight="bold",
                    ha="center", va="center", zorder=5)
            
            # Text inside Card
            ax.text(x + 5.5, y_pos + 6.0, item, color=c_primary, fontsize=8.5, fontweight="medium",
                    ha="left", va="center", fontfamily="sans-serif", wrap=True, multialignment="left")

    # Connective Inter-Stage Arrows
    for i in range(len(stages) - 1):
        x_start = stages[i]["x"] + stages[i]["w"]
        x_end = stages[i+1]["x"]
        y_mid = 48
        
        arrow = patches.FancyArrowPatch((x_start + 0.2, y_mid), (x_end - 0.2, y_mid),
                                        arrowstyle="simple,head_width=5,head_length=6",
                                        color=c_blue, linewidth=1.5, zorder=5)
        ax.add_patch(arrow)

    # Bottom Footer Banner
    footer_box = FancyBboxPatch((2, 2.5), 96, 5.5, boxstyle="round,pad=0.4,rounding_size=0.8",
                                facecolor="#0F172A", edgecolor="none", zorder=2)
    ax.add_patch(footer_box)
    
    ax.text(50, 5.8, "★ Rigorous Medical AI Standards: Nested CV • Ablation Validated • Counterfactual Actionability • DeLong Verified",
            color="#38BDF8", fontsize=9.5, fontweight="bold", ha="center", va="center", fontfamily="sans-serif")
    ax.text(50, 3.8, "Proposed for Submission to: Computers in Biology and Medicine / IEEE JBHI / Nature Scientific Reports",
            color="#94A3B8", fontsize=8.5, ha="center", va="center", fontfamily="sans-serif")

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, facecolor=bg_color, bbox_inches="tight")
    plt.savefig(svg_path, format="svg", facecolor=bg_color, bbox_inches="tight")
    plt.close()
    print(f"Diagram successfully generated at:\n  PNG: {png_path}\n  SVG: {svg_path}")

if __name__ == "__main__":
    create_publication_workflow()
