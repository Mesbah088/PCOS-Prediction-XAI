import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def create_upgraded_workflow_diagram():
    output_dir = r"c:\Users\user\OneDrive\Desktop\AI_ML\PCOS-Prediction-XAI\outputs"
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "upgraded_workflow_comparison.png")
    svg_path = os.path.join(output_dir, "upgraded_workflow_comparison.svg")

    # High Resolution Canvas matching user's layout structure
    fig = plt.figure(figsize=(16, 11), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    bg_color = "#FFFFFF"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Color Palette for Upgrades
    c_base_blue = "#0284C7"       # Primary header blue
    c_box_stroke = "#64748B"      # Slate dashed border
    c_card_bg = "#F8FAFC"         # Card background
    c_upgrade_green = "#059669"   # Green badge for upgrades
    c_chip_bg = "#EFF6FF"         # Chip light blue
    c_chip_border = "#93C5FD"     # Chip border
    c_new_chip_bg = "#ECFDF5"     # Chip light green
    c_new_chip_border = "#6EE7B7" # Chip green border

    # Main Top Dataset Node
    ds_box = FancyBboxPatch((13, 91), 14, 5.5, boxstyle="round,pad=0.3,rounding_size=0.6",
                            facecolor=c_base_blue, edgecolor="none", zorder=3)
    ax.add_patch(ds_box)
    ax.text(20, 93.75, "PCOS Dataset\n(541 Patients, 44 Feats)", color="#FFFFFF", fontsize=9, fontweight="bold",
            ha="center", va="center", multialignment="center")

    # Down arrow from Dataset to Pre-processing
    ax.annotate("", xy=(20, 83.5), xytext=(20, 91),
                arrowprops=dict(facecolor="#0F172A", edgecolor="#0F172A", width=1.5, headwidth=6, headlength=6))

    # ==========================================
    # 1. PRE-PROCESSING BOX (Top Left)
    # ==========================================
    pre_outer = FancyBboxPatch((4, 53), 32, 30, boxstyle="round,pad=0.5,rounding_size=1.2",
                               facecolor=c_card_bg, edgecolor=c_box_stroke, linestyle="--", linewidth=1.5, zorder=2)
    ax.add_patch(pre_outer)
    ax.text(20, 80.5, "1. Pre-processing & Feature Eng.", color="#0F172A", fontsize=11, fontweight="bold", ha="center")
    
    # Sub items
    pre_items = [
        ("Missing Value & Outlier Imputation", c_chip_bg, c_chip_border, "#1E293B"),
        ("Robust & Quantile Normalization", c_chip_bg, c_chip_border, "#1E293B"),
        ("★ REPLACED/ADDED: 22 Rotterdam Bio-Markers\n(PCOM Flag, LH/FSH Ratio, Androgen Score)", c_new_chip_bg, c_new_chip_border, "#065F46")
    ]
    y_p = 73.5
    for text, bg, brd, txt_col in pre_items:
        h = 6.5 if "\n" in text else 4.2
        c = FancyBboxPatch((6, y_p - h/2), 28, h, boxstyle="round,pad=0.2,rounding_size=0.5",
                           facecolor=bg, edgecolor=brd, linewidth=1.2, zorder=3)
        ax.add_patch(c)
        ax.text(20, y_p, text, color=txt_col, fontsize=7.8, fontweight="bold" if "★" in text else "normal",
                ha="center", va="center", multialignment="center")
        y_p -= (h + 2.2)

    # Down arrow from Preprocessing to Data Splitting
    ax.annotate("", xy=(20, 45.5), xytext=(20, 53),
                arrowprops=dict(facecolor="#0F172A", edgecolor="#0F172A", width=1.5, headwidth=6, headlength=6))

    # ==========================================
    # 2. DATA SPLITTING BOX (Bottom Left)
    # ==========================================
    split_outer = FancyBboxPatch((4, 18), 32, 27, boxstyle="round,pad=0.5,rounding_size=1.2",
                                 facecolor=c_card_bg, edgecolor=c_box_stroke, linestyle="--", linewidth=1.5, zorder=2)
    ax.add_patch(split_outer)
    ax.text(20, 42.0, "2. Data Splitting & Balancing", color="#0F172A", fontsize=11, fontweight="bold", ha="center")

    # Split sub-boxes
    s_test = FancyBboxPatch((6, 33), 13, 5.5, boxstyle="round,pad=0.2,rounding_size=0.4",
                            facecolor="#E2E8F0", edgecolor="#94A3B8", linewidth=1.0, zorder=3)
    ax.add_patch(s_test)
    ax.text(12.5, 35.75, "Validation Set\n(Untouched)", color="#1E293B", fontsize=7.5, fontweight="bold", ha="center", va="center")

    s_train = FancyBboxPatch((21, 33), 13, 5.5, boxstyle="round,pad=0.2,rounding_size=0.4",
                             facecolor="#E2E8F0", edgecolor="#94A3B8", linewidth=1.0, zorder=3)
    ax.add_patch(s_train)
    ax.text(27.5, 35.75, "Training Set\n(Cross-Val)", color="#1E293B", fontsize=7.5, fontweight="bold", ha="center", va="center")

    # SMOTE upgrade box
    smote_box = FancyBboxPatch((6, 21), 28, 8.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                               facecolor=c_new_chip_bg, edgecolor=c_new_chip_border, linewidth=1.2, zorder=3)
    ax.add_patch(smote_box)
    ax.text(20, 25.25, "★ REPLACED: Nested 5x5 CV + SMOTE-NC\n(Inside-Fold Balancing • Zero Data Leakage)",
            color="#065F46", fontsize=7.5, fontweight="bold", ha="center", va="center", multialignment="center")

    # Arrow from Splitting to Feature Selection (Up & Right)
    ax.annotate("", xy=(38, 70), xytext=(36, 26),
                arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=8",
                                color="#0F172A", lw=1.8))

    # ==========================================
    # 3. FEATURE SELECTION BOX (Top Middle)
    # ==========================================
    fs_outer = FancyBboxPatch((40, 53), 26, 40, boxstyle="round,pad=0.5,rounding_size=1.2",
                              facecolor=c_card_bg, edgecolor=c_box_stroke, linestyle="--", linewidth=1.5, zorder=2)
    ax.add_patch(fs_outer)
    ax.text(53, 90.0, "3. Tri-Stage Hybrid Selection", color="#0F172A", fontsize=11, fontweight="bold", ha="center")
    
    fs_cards = [
        ("Stage 1: Chi2 & ANOVA Filter", "44 -> 30 Features", c_chip_bg, c_chip_border),
        ("Stage 2: Boruta + CatBoost RFE", "30 -> 18 Non-linear Features", c_chip_bg, c_chip_border),
        ("★ Stage 3: ElasticNet Stability", "18 -> Final 16 Critical Bio-Markers", c_new_chip_bg, c_new_chip_border)
    ]
    y_fs = 82.5
    for title, desc, bg, brd in fs_cards:
        c = FancyBboxPatch((42, y_fs - 4.5), 22, 7.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                           facecolor=bg, edgecolor=brd, linewidth=1.2, zorder=3)
        ax.add_patch(c)
        ax.text(53, y_fs - 0.5, title, color="#0F172A", fontsize=8.0, fontweight="bold", ha="center")
        ax.text(53, y_fs - 3.2, desc, color="#047857" if "★" in title else "#0284C7", fontsize=7.2, fontweight="bold", ha="center")
        y_fs -= 10.5

    # Down arrow from Feature Selection to ML Models
    ax.annotate("", xy=(53, 44), xytext=(53, 53),
                arrowprops=dict(facecolor="#0F172A", edgecolor="#0F172A", width=1.5, headwidth=6, headlength=6))

    # ==========================================
    # 4. ML MODELS BENCHMARK BOX (Bottom Middle)
    # ==========================================
    ml_outer = FancyBboxPatch((40, 10), 26, 33, boxstyle="round,pad=0.5,rounding_size=1.2",
                              facecolor=c_card_bg, edgecolor=c_box_stroke, linestyle="--", linewidth=1.5, zorder=2)
    ax.add_patch(ml_outer)
    ax.text(53, 40.5, "4. Multi-Model Benchmark", color="#0F172A", fontsize=11, fontweight="bold", ha="center")

    models_list = [
        ["XGBoost", "CatBoost", "LightGBM ★"],
        ["Google TabNet ★", "Deep ResNet ★", "MLP"],
        ["Random Forest", "SVM (RBF)", "LogReg (L2)"]
    ]
    for row_idx, row in enumerate(models_list):
        for col_idx, m_name in enumerate(row):
            x_m = 42 + (col_idx * 7.4)
            y_m = 32.5 - (row_idx * 7.5)
            is_up = "★" in m_name
            m_card = FancyBboxPatch((x_m, y_m), 6.8, 5.5, boxstyle="round,pad=0.1,rounding_size=0.3",
                                    facecolor=c_new_chip_bg if is_up else "#FEE2E2",
                                    edgecolor=c_new_chip_border if is_up else "#FCA5A5", linewidth=1.0, zorder=3)
            ax.add_patch(m_card)
            ax.text(x_m + 3.4, y_m + 2.75, m_name, color="#065F46" if is_up else "#991B1B",
                    fontsize=6.5, fontweight="bold", ha="center", va="center")

    ax.text(53, 12.5, "Bayesian Hyper-Tuning (Optuna Search)", color="#475569", fontsize=7.2, fontweight="bold", ha="center")

    # Arrow from ML Models to Voting (Right & Down)
    ax.annotate("", xy=(70, 20), xytext=(66, 20),
                arrowprops=dict(facecolor="#0F172A", edgecolor="#0F172A", width=1.5, headwidth=6, headlength=6))

    # ==========================================
    # 5. VOTING & ENSEMBLE BOX (Bottom Right)
    # ==========================================
    vote_outer = FancyBboxPatch((70, 5), 26, 28, boxstyle="round,pad=0.5,rounding_size=1.2",
                                facecolor=c_card_bg, edgecolor=c_box_stroke, linestyle="--", linewidth=1.5, zorder=2)
    ax.add_patch(vote_outer)
    ax.text(83, 30.5, "5. Super-Ensemble & Stacking", color="#0F172A", fontsize=10.5, fontweight="bold", ha="center")

    v_card = FancyBboxPatch((72, 8), 22, 19.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                            facecolor=c_new_chip_bg, edgecolor=c_new_chip_border, linewidth=1.3, zorder=3)
    ax.add_patch(v_card)
    ax.text(83, 24.5, "★ REPLACED: Neuro-Tree Stacking", color="#065F46", fontsize=8.2, fontweight="bold", ha="center")
    ax.text(83, 19.5, "TabNet + ResNet + LGBM + CB + XGB", color="#1E293B", fontsize=7.2, fontweight="bold", ha="center")
    ax.text(83, 14.8, "Dynamic Soft-Voting Meta Learner", color="#475569", fontsize=6.8, ha="center")
    ax.text(83, 10.5, "Platt Scaling Calibration (Brier <= 0.04)", color="#047857", fontsize=6.8, fontweight="bold", ha="center")

    # Up arrow from Voting to Model Evaluation
    ax.annotate("", xy=(83, 37.5), xytext=(83, 33),
                arrowprops=dict(facecolor="#0F172A", edgecolor="#0F172A", width=1.5, headwidth=6, headlength=6))

    # ==========================================
    # 6. MODEL EVALUATION BOX (Middle Right)
    # ==========================================
    eval_outer = FancyBboxPatch((70, 37.5), 26, 26, boxstyle="round,pad=0.5,rounding_size=1.2",
                                facecolor=c_card_bg, edgecolor=c_box_stroke, linestyle="--", linewidth=1.5, zorder=2)
    ax.add_patch(eval_outer)
    ax.text(83, 60.5, "6. Model Evaluation", color="#0F172A", fontsize=11, fontweight="bold", ha="center")

    metrics = ["Accuracy: 94.5% - 97.0%", "ROC-AUC: 0.9726 ★", "Precision: 96.88% ★", "Sensitivity: 91.67% ★"]
    for i, m in enumerate(metrics):
        x_ev = 72 + (i % 2) * 11.2
        y_ev = 52.5 - (i // 2) * 6.5
        is_up = "★" in m
        m_c = FancyBboxPatch((x_ev, y_ev), 10.5, 5.0, boxstyle="round,pad=0.1,rounding_size=0.3",
                             facecolor=c_new_chip_bg if is_up else "#FEF3C7",
                             edgecolor=c_new_chip_border if is_up else "#FDE68A", linewidth=1.0, zorder=3)
        ax.add_patch(m_c)
        ax.text(x_ev + 5.25, y_ev + 2.5, m, color="#065F46" if is_up else "#92400E",
                fontsize=6.8, fontweight="bold", ha="center", va="center")

    ax.text(83, 41.5, "★ DeLong AUC Test & Wilcoxon (p < 0.001)", color="#059669", fontsize=7.2, fontweight="bold", ha="center")

    # Up arrow from Evaluation to XAI
    ax.annotate("", xy=(83, 68), xytext=(83, 63.5),
                arrowprops=dict(facecolor="#0F172A", edgecolor="#0F172A", width=1.5, headwidth=6, headlength=6))

    # ==========================================
    # 7. EXPLAINABLE AI & DEPLOYMENT (Top Right)
    # ==========================================
    xai_outer = FancyBboxPatch((70, 68), 26, 25, boxstyle="round,pad=0.5,rounding_size=1.2",
                               facecolor=c_card_bg, edgecolor=c_box_stroke, linestyle="--", linewidth=1.5, zorder=2)
    ax.add_patch(xai_outer)
    ax.text(83, 90.0, "7. 360° XAI & CDSS Deployment", color="#0F172A", fontsize=10.5, fontweight="bold", ha="center")

    xai_cards = [
        ("Global TreeSHAP & Local LIME", "#E2E8F0", "#94A3B8", "#1E293B"),
        ("★ DiCE Counterfactual Prescriptions\n(Personalized Patient Action Guidance)", c_new_chip_bg, c_new_chip_border, "#065F46"),
        ("★ Streamlit Web CDSS App + Nomogram", c_new_chip_bg, c_new_chip_border, "#065F46")
    ]
    y_x = 83.5
    for text, bg, brd, txt_col in xai_cards:
        h = 5.8 if "\n" in text else 4.0
        c = FancyBboxPatch((72, y_x - h/2), 22, h, boxstyle="round,pad=0.2,rounding_size=0.4",
                           facecolor=bg, edgecolor=brd, linewidth=1.1, zorder=3)
        ax.add_patch(c)
        ax.text(83, y_x, text, color=txt_col, fontsize=7.0, fontweight="bold" if "★" in text else "normal",
                ha="center", va="center", multialignment="center")
        y_x -= (h + 1.8)

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, facecolor=bg_color, bbox_inches="tight")
    plt.savefig(svg_path, format="svg", facecolor=bg_color, bbox_inches="tight")
    plt.close()
    print(f"Upgraded comparison workflow generated at:\n  PNG: {png_path}\n  SVG: {svg_path}")

if __name__ == "__main__":
    create_upgraded_workflow_diagram()
