import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (on all pages except page 1)
        if self._pageNumber > 1:
            self.drawString(40, 810, "PCOS AI RESEARCH • COMPLETE TECHNOLOGIES & METHODOLOGY STUDY GUIDE")
            self.drawRightString(555, 810, "Master's Thesis & Viva Preparation")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(40, 804, 555, 804)

        # Footer (on all pages)
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(555, 26, page_text)
        self.drawString(40, 26, "PCOS Prediction & Explainable AI Framework | Comprehensive Technology Reference")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(40, 36, 555, 36)
        self.restoreState()

def build_pdf_study_guide(pdf_filename: str):
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=A4,
        leftMargin=38,
        rightMargin=38,
        topMargin=42,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Normal'], fontName='Helvetica-Bold',
        fontSize=15, leading=19, textColor=colors.HexColor("#0F172A"), spaceAfter=3
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle', parent=styles['Normal'], fontName='Helvetica',
        fontSize=9.5, leading=13.5, textColor=colors.HexColor("#334155"), spaceAfter=4
    )

    meta_style = ParagraphStyle(
        'DocMeta', parent=styles['Normal'], fontName='Helvetica-Bold',
        fontSize=8.5, leading=11, textColor=colors.HexColor("#0284C7")
    )

    h1_style = ParagraphStyle(
        'H1_Custom', parent=styles['Normal'], fontName='Helvetica-Bold',
        fontSize=11.5, leading=15, textColor=colors.HexColor("#0F172A"),
        spaceBefore=7, spaceAfter=3, keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'H2_Custom', parent=styles['Normal'], fontName='Helvetica-Bold',
        fontSize=9.5, leading=12.5, textColor=colors.HexColor("#0369A1"),
        spaceBefore=4, spaceAfter=2, keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom', parent=styles['Normal'], fontName='Helvetica',
        fontSize=8.8, leading=12.5, textColor=colors.HexColor("#334155"), spaceAfter=3
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom', parent=styles['Normal'], fontName='Helvetica',
        fontSize=8.6, leading=12.2, textColor=colors.HexColor("#334155"),
        leftIndent=8, firstLineIndent=-5, spaceAfter=2.5
    )

    code_style = ParagraphStyle(
        'Code_Custom', parent=styles['Normal'], fontName='Courier',
        fontSize=7.8, leading=10.2, textColor=colors.HexColor("#0F172A")
    )

    qa_question = ParagraphStyle(
        'QA_Q', parent=styles['Normal'], fontName='Helvetica-Bold',
        fontSize=8.8, leading=12, textColor=colors.HexColor("#B91C1C")
    )

    qa_answer = ParagraphStyle(
        'QA_A', parent=styles['Normal'], fontName='Helvetica',
        fontSize=8.6, leading=12, textColor=colors.HexColor("#1E293B")
    )

    table_header = ParagraphStyle(
        'TableHeader', parent=styles['Normal'], fontName='Helvetica-Bold',
        fontSize=7.8, leading=10, textColor=colors.white
    )

    table_cell = ParagraphStyle(
        'TableCell', parent=styles['Normal'], fontName='Helvetica',
        fontSize=7.4, leading=9.4, textColor=colors.HexColor("#1E293B")
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold', parent=styles['Normal'], fontName='Helvetica-Bold',
        fontSize=7.4, leading=9.4, textColor=colors.HexColor("#0F172A")
    )

    story = []

    # Title Banner
    story.append(Paragraph("TECHNICAL REFERENCE & COMPREHENSIVE STUDY GUIDE", meta_style))
    story.append(Paragraph("PCOS Machine Learning & Explainable AI: Complete Technology Guide", title_style))
    story.append(Paragraph("In-depth Exploration of 13 Technologies, Clinical Rationales, Algorithms & Thesis Defense Q&A", subtitle_style))
    story.append(Paragraph("<b>Scope:</b> Data Preprocessing, Boundary Balancing, 15-Classifier Ensemble, XAI Suite & Clinical CDSS", meta_style))
    story.append(Spacer(1, 2))
    story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#0284C7"), spaceBefore=2, spaceAfter=5))

    # Overview Callout
    overview_text = (
        "<b>Study Guide Purpose:</b> This document provides an exhaustive, crystal-clear explanation of every tool, library, "
        "and biomedical concept utilized to construct the multimodal PCOS prediction and explainable decision framework. "
        "Each section outlines (1) what the technology is, (2) its specific scientific purpose in our pipeline, "
        "(3) architectural implementation details, and (4) essential Viva / Thesis Defense Q&A answers."
    )
    callout = Table([[Paragraph(overview_text, body_style)]], colWidths=[519])
    callout.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F0F9FF")),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor("#BAE6FD")),
        ('LINELEFT', (0, 0), (0, -1), 3, colors.HexColor("#0284C7")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(callout)
    story.append(Spacer(1, 4))

    # Section 1
    story.append(Paragraph("1. Python 3.13 & Virtual Environments (Execution Foundation)", h1_style))
    story.append(Paragraph("• <b>Concept:</b> Python is the primary programming language for modern scientific computing, while <code>virtualenv</code> provides an isolated runtime containing precise library versions.", bullet_style))
    story.append(Paragraph("• <b>Role in Research:</b> Resolves package conflicts between cutting-edge ML boosters (LightGBM, CatBoost) and complex explainability engines (SHAP, DiCE). Guarantees 100% deterministic reproducibility across platforms.", bullet_style))
    story.append(Paragraph("• <b>Viva Defense Q&A:</b> <i>\"How do you ensure independent researchers can reproduce your findings?\"</i> &rarr; <b>Answer:</b> By freezing random seeds (<code>random_state=42</code>) across all estimators, standardizing the Python 3.13 environment, and packaging all dependencies in <code>requirements.txt</code>.", bullet_style))
    story.append(Spacer(1, 4))

    # Section 2
    story.append(Paragraph("2. Pandas & NumPy (Data Wrangling & Biomarker Engineering)", h1_style))
    story.append(Paragraph("• <b>Concept:</b> Pandas provides columnar DataFrame structures for tabular medical data; NumPy accelerates array operations via C-optimized vectorized linear algebra.", bullet_style))
    story.append(Paragraph("• <b>Role in Research:</b> Cleaned the 541-patient Kerala Hospital cohort, imputed missing data inside cross-validation folds, and engineered <b>25 novel clinical interaction biomarkers</b>.", bullet_style))
    story.append(Paragraph("• <b>Key Biomarker Formulas:</b>", bullet_style))
    story.append(Paragraph("  - <i>Follicle Geometric Mean:</i> <code>sqrt(Follicle_No._L &times; Follicle_No._R)</code> (penalizes extreme unilateral variance).", bullet_style))
    story.append(Paragraph("  - <i>Skin Metabolic Score:</i> <code>hair_growth + Skin_darkening + Pimples + Weight_gain</code> (quantifies systemic androgenism).", bullet_style))
    story.append(Spacer(1, 4))

    # Section 3
    story.append(Paragraph("3. Rotterdam Consensus Criteria (Biomedical Ground Truth)", h1_style))
    story.append(Paragraph("• <b>Concept:</b> The internationally recognized diagnostic standard (Rotterdam 2003 / 2023 update). Requires at least 2 of 3 clinical phenotypes: (1) Oligo/anovulation, (2) Hyperandrogenism, and (3) Polycystic Ovarian Morphology (PCOM &ge; 12 follicles in 2003; &ge; 20 in 2023).", bullet_style))
    story.append(Paragraph("• <b>Clinical Integration:</b> Guided the creation of <code>PCOM_Severe_Flag</code> (&ge; 20 follicles) and <code>PCOM_Rotterdam_Flag</code> (&ge; 12 follicles). These two engineered features ranked <b>#1 and #2</b> in diagnostic importance across all models, proving strict biological convergence.", bullet_style))
    story.append(Spacer(1, 4))

    # Section 4
    story.append(Paragraph("4. Imbalanced-Learn & SMOTE-ENN (Boundary Cleansing & Zero-Leakage)", h1_style))
    story.append(Paragraph("• <b>The Imbalance Challenge:</b> The cohort contains 364 healthy controls and 177 PCOS patients (~2:1 imbalance). Conventional SMOTE creates synthetic points blindly, generating ambiguous samples deep within control clusters.", bullet_style))
    story.append(Paragraph("• <b>The Solution (SMOTE-ENN):</b> A two-phase hybrid method. Phase 1 (SMOTE) interpolates synthetic minority instances; Phase 2 (Edited Nearest Neighbors - ENN) prunes ambiguous borderline points that overlap majority boundaries, cleaning decision margins.", bullet_style))
    story.append(Paragraph("• <b>Zero-Leakage Rule:</b> Resampling was fitted <b>strictly within training folds</b> ($432 &rarr; 468$ balanced instances). The 109-patient held-out test cohort was never resampled, maintaining pristine clinical validity.", bullet_style))
    story.append(Spacer(1, 4))

    # Page Break for Page 2
    story.append(PageBreak())

    # Section 5
    story.append(Paragraph("5. Scikit-Learn: Machine Learning Pipeline & 9 Baseline Architectures", h1_style))
    story.append(Paragraph("• <b>Concept:</b> Scikit-learn provides standardized transformer pipelines and model benchmarking infrastructure.", bullet_style))
    story.append(Paragraph("• <b>Components Implemented:</b>", bullet_style))
    story.append(Paragraph("  - <code>MinMaxScaler</code>: Normalizes feature distributions strictly inside folds.", bullet_style))
    story.append(Paragraph("  - <code>RepeatedStratifiedKFold</code>: 5-Fold Cross-Validation with stratified class representation.", bullet_style))
    story.append(Paragraph("  - <b>9 Diverse Baselines:</b> Random Forest, Extra Trees, Gradient Boosting, MLP Neural Net (64-32 ReLU units), Support Vector Machine (RBF kernel, C=1.5), Logistic Regression, LDA, Gaussian Naive Bayes, and KNN (k=5).", bullet_style))
    story.append(Spacer(1, 4))

    # Section 6
    story.append(Paragraph("6. State-of-the-Art Gradient Boosters (LightGBM, CatBoost, XGBoost)", h1_style))
    story.append(Paragraph("• <b>Why Tree Boosters?</b> Tabular clinical datasets with complex biomarker interactions are best modeled by gradient boosted decision trees (GBDTs). We harnessed three distinct architectural paradigms:", bullet_style))
    
    gbdt_table_data = [
        [Paragraph("Architecture", table_header), Paragraph("Tree Topology", table_header), Paragraph("Key Advantage in PCOS Research", table_header)],
        [Paragraph("<b>LightGBM</b>", table_cell_bold), Paragraph("Leaf-wise (Best-First) Growth", table_cell), Paragraph("Discovers deep non-linear multi-biomarker splits with unmatched computational speed.", table_cell)],
        [Paragraph("<b>CatBoost</b>", table_cell_bold), Paragraph("Symmetric (Oblivious) Trees", table_cell), Paragraph("Prevents overfitting on small medical cohorts and handles binary symptom flags flawlessly.", table_cell)],
        [Paragraph("<b>XGBoost</b>", table_cell_bold), Paragraph("Depth-wise Greedy Splitting", table_cell), Paragraph("Robust L1/L2 shrinkage controlling variance and stabilizing leaf weight estimates.", table_cell)],
    ]
    gbdt_table = Table(gbdt_table_data, colWidths=[90, 140, 289])
    gbdt_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(gbdt_table)
    story.append(Spacer(1, 5))

    # Section 7
    story.append(Paragraph("7. Super-Ensemble & Stacking Meta-Learning (The Champions)", h1_style))
    story.append(Paragraph("• <b>Champion Soft-Voting Super-Ensemble:</b> Fuses posterior class probabilities from 4 top diverse paradigms:", bullet_style))
    story.append(Paragraph("  <code>P(PCOS) = [1.2 &middot; P(LGBM) + 1.2 &middot; P(CatBoost) + 1.0 &middot; P(XGBoost) + 0.8 &middot; P(MLP)] / 4.2</code>", code_style))
    story.append(Paragraph("  - <i>Why it succeeded:</i> Neural net (MLP) errors are uncorrelated with tree booster errors. Combining them reduced missed cases to <b>only 4 out of 36 PCOS patients (88.89% Recall)</b> and reached <b>98.36 ± 1.04% 5-Fold CV F1</b>.", bullet_style))
    story.append(Paragraph("• <b>Champion Stacking Meta-Learner:</b> Uses out-of-fold predictions from Level-0 estimators to train a Level-1 $L_2$-regularized Logistic Regression meta-classifier, reaching <b>91.74% Accuracy, 93.15% Specificity, and 0.9441 ROC-AUC</b>.", bullet_style))
    story.append(Spacer(1, 4))

    # Section 8
    story.append(Paragraph("8. Platt Scaling Probability Calibration (Clinical Confidence Alignment)", h1_style))
    story.append(Paragraph("• <b>The Problem:</b> Uncalibrated decision trees produce overconfident probabilities clustering near 0 and 1 due to high leaf purity. In clinical triage, physicians require dependable risk probabilities.", bullet_style))
    story.append(Paragraph("• <b>The Method:</b> Post-hoc <b>Platt Scaling</b> fits a univariate sigmoid logistic regression on out-of-fold ensemble margins via 5-fold cross-validation.", bullet_style))
    story.append(Paragraph("• <b>Empirical Result:</b> Brier Score improved to <b>0.0633 / 0.0672</b>, perfectly aligning predicted probability with true disease prevalence.", bullet_style))
    story.append(Spacer(1, 4))

    # Section 9
    story.append(Paragraph("9. SciPy: Wilcoxon Signed-Rank Test (Statistical Superiority Proof)", h1_style))
    story.append(Paragraph("• <b>Concept:</b> Non-parametric paired hypothesis test comparing performance across cross-validation folds.", bullet_style))
    story.append(Paragraph("• <b>Result:</b> Confirms that the Champion Ensemble's superiority over single models is statistically significant ($p < 0.05$) and not due to chance dataset partitioning.", bullet_style))
    story.append(Spacer(1, 4))

    # Page Break for Page 3
    story.append(PageBreak())

    # Section 10
    story.append(Paragraph("10. 360° Explainable AI (XAI) Suite (Eliminating the Black Box)", h1_style))
    story.append(Paragraph("To meet clinical decision support standards (FDA/CE SaMD guidelines), we deployed a tripartite XAI architecture:", body_style))

    xai_table_data = [
        [Paragraph("XAI Framework", table_header), Paragraph("Underlying Math", table_header), Paragraph("Clinical Question Answered", table_header)],
        [Paragraph("<b>SHAP</b><br/>(TreeExplainer)", table_cell_bold), Paragraph("Game-theoretic Shapley Values (Additive Attribution)", table_cell), Paragraph("<i>\"Which biomarkers contributed most to this patient's high risk score?\"</i><br/>Produces global beeswarm and local waterfall breakdowns.", table_cell)],
        [Paragraph("<b>LIME</b><br/>(Tabular Explainer)", table_cell_bold), Paragraph("Sparse Local Linear Surrogate Approximations", table_cell), Paragraph("<i>\"What explicit clinical threshold rules triggered the diagnosis?\"</i><br/>e.g., Follicle_No._R > 10 AND PCOM_Rotterdam_Flag = 1.", table_cell)],
        [Paragraph("<b>DiCE</b><br/>(Diverse Counterfactuals)", table_cell_bold), Paragraph("Gradient Loss Optimization: Min Feature Perturbation", table_cell), Paragraph("<i>\"How can this patient reverse their condition?\"</i><br/>e.g., Reduce Weight to 64 kg and cease fast food to drop risk from 89% to 22%.", table_cell)],
        [Paragraph("<b>Clinical Nomogram</b><br/>(Custom Algorithm)", table_cell_bold), Paragraph("Log-Odds Point Mapping to Linear Risk Scales", table_cell), Paragraph("<i>\"Can a doctor calculate risk at the bedside without a computer?\"</i><br/>Paper-based ruler scoring system using biomarker points.", table_cell)],
    ]
    xai_table = Table(xai_table_data, colWidths=[95, 135, 289])
    xai_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(xai_table)
    story.append(Spacer(1, 5))

    # Section 11 & 12
    story.append(Paragraph("11. Joblib (Model Serialization & Clinical Preservation)", h1_style))
    story.append(Paragraph("• <b>Implementation:</b> Encapsulated the entire pipeline (scalers, encoders, weights of 4 models, Platt sigmoid mappings) into <code>outputs/champion_ensemble_model.joblib</code> (2.76 MB). Enables instantaneous zero-latency inference (< 10 ms).", bullet_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("12. Streamlit (Interactive Clinical Decision Support System - CDSS)", h1_style))
    story.append(Paragraph("• <b>Implementation (`app.py`):</b> Full web application allowing doctors to enter patient clinical values, receiving real-time risk gauges (Green/Orange/Red), instant SHAP waterfalls, and actionable DiCE lifestyle prescriptions.", bullet_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("13. ReportLab (Academic Publication-Quality PDF Engine)", h1_style))
    story.append(Paragraph("• <b>Implementation:</b> Programmatic vector graphics engine producing pixel-perfect A4 reports with exact margins, headers, dynamic canvas page numbers, and embedded diagnostic graphics for thesis defense.", bullet_style))
    story.append(Spacer(1, 5))

    # Section 14: Defense Q&A Box
    story.append(Paragraph("14. Master's Thesis Defense & Viva Q&A Master Cheat Sheet", h1_style))
    
    qa_data = [
        [Paragraph("Q1: Why did you prioritize Sensitivity / Recall over Accuracy?", qa_question)],
        [Paragraph("<b>Answer:</b> In medical screening, <b>False Negatives (missed diagnoses)</b> carry severe health risks. An undetected PCOS patient misses early interventions for insulin resistance, type 2 diabetes, and endometrial cancer. Our Champion Ensemble detected 32 of 36 positive patients (88.89% recall), missing only 4 cases.", qa_answer)],
        [Paragraph("Q2: What is the scientific justification for engineering 25 Rotterdam features?", qa_question)],
        [Paragraph("<b>Answer:</b> Raw biological markers operate through non-linear endocrine interactions. For instance, the ratio of follicles to age or the geometric mean of bilateral follicle counts captures ovarian morphological density much more effectively than raw numbers alone. 9 of the final 16 invariant features were domain-engineered, proving their superior predictive utility.", qa_answer)],
        [Paragraph("Q3: Why not rely solely on deep learning / neural networks?", qa_question)],
        [Paragraph("<b>Answer:</b> Deep learning requires massive training samples to avoid overfitting on tabular data. Tree ensembles (LightGBM, CatBoost, XGBoost) consistently outperform neural networks on tabular cohorts. However, blending a 2-layer MLP with tree boosters in our Soft-Voting Ensemble gave the best of both worlds, lowering overall prediction variance.", qa_answer)],
        [Paragraph("Q4: How does DiCE counterfactuals differ from SHAP explanations?", qa_question)],
        [Paragraph("<b>Answer:</b> SHAP is retrospective (explaining <i>why</i> a patient was categorized as high-risk). DiCE is prospective and prescriptive (identifying the minimal, feasible lifestyle modifications a patient must execute to transition from High Risk to Low Risk).", qa_answer)],
    ]
    qa_table = Table(qa_data, colWidths=[519])
    qa_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FEF2F2")),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor("#FECACA")),
        ('LINELEFT', (0, 0), (0, -1), 3, colors.HexColor("#DC2626")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(qa_table)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Technology Study Guide PDF successfully generated at: {pdf_filename}")

if __name__ == "__main__":
    out_pdf = os.path.join(os.path.dirname(__file__), "..", "PCOS_Technologies_and_Methodology_Study_Guide.pdf")
    build_pdf_study_guide(out_pdf)
