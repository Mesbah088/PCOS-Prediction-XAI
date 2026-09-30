import pandas as pd
import numpy as np
from typing import Tuple

def engineer_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers novel domain-informed endocrinological, ultrasound, and metabolic features
    aligned with international Rotterdam consensus diagnostic criteria:
    
    1. Ultrasound & Morphological Markers:
       - Total Follicle Count = Follicle (L) + Follicle (R)
       - Follicle Difference & Asymmetry Index
       - Geometric Mean Follicle Count = sqrt(Follicle L * Follicle R)
       - Rotterdam PCOM Binary Flag: Follicle Count >= 10 in either ovary
       - Severe PCOM Flag: Follicle Count >= 12 in either ovary
       
    2. Endocrine & Hormonal Biomarkers:
       - LH / FSH Ratio & Non-Linear Log Ratio
       - Clinical High LH/FSH Flag (> 2.0)
       - AMH * Total Follicles Interaction & AMH Log Transform
       
    3. Anthropometric & Metabolic Indicators:
       - Calculated BMI & Overweight Flag (BMI >= 25)
       - Waist-to-Hip Ratio (WHR) & Android Obesity Flag (WHR >= 0.85)
       - Hirsutism & Hyperandrogenism Composite Score
       - Skin Acanthosis & Metabolic Cluster Index
       - Total Metabolic Symptom Score (sum of all 6 clinical physical indicators)
       
    4. Ovarian Dynamics & Reserve:
       - Follicles Per Year of Age (Ovarian Reserve Density)
       - Endometrial Thickness to Cycle Length Ratio
       - Menstrual Cycle Severity Index
    """
    df_out = df.copy()

    # 1. Ultrasound & Follicle Metrics
    if 'Follicle No. (L)' in df_out.columns and 'Follicle No. (R)' in df_out.columns:
        f_l = df_out['Follicle No. (L)']
        f_r = df_out['Follicle No. (R)']
        
        df_out['Total_Follicles'] = f_l + f_r
        df_out['Follicle_Diff'] = (f_l - f_r).abs()
        df_out['Follicle_Asymmetry_Index'] = df_out['Follicle_Diff'] / (df_out['Total_Follicles'] + 1e-5)
        df_out['Follicle_Geometric_Mean'] = np.sqrt(np.maximum(0, f_l) * np.maximum(0, f_r))
        df_out['PCOM_Rotterdam_Flag'] = ((f_l >= 10) | (f_r >= 10)).astype(int)
        df_out['PCOM_Severe_Flag'] = ((f_l >= 12) | (f_r >= 12)).astype(int)
        
        if 'Avg. F size (L) (mm)' in df_out.columns and 'Avg. F size (R) (mm)' in df_out.columns:
            df_out['Mean_Follicle_Size_Both'] = (df_out['Avg. F size (L) (mm)'] + df_out['Avg. F size (R) (mm)']) / 2.0

    # 2. Hormonal & Endocrine Ratios
    if 'LH(mIU/mL)' in df_out.columns and 'FSH(mIU/mL)' in df_out.columns:
        fsh_safe = df_out['FSH(mIU/mL)'].replace(0, 0.01)
        lh_val = np.maximum(0, df_out['LH(mIU/mL)'])
        fsh_val = np.maximum(0, df_out['FSH(mIU/mL)'])
        
        df_out['LH_FSH_Ratio_Calculated'] = df_out['LH(mIU/mL)'] / fsh_safe
        df_out['High_LH_FSH_Flag'] = (df_out['LH_FSH_Ratio_Calculated'] > 2.0).astype(int)
        df_out['LH_FSH_Log_Ratio'] = np.log1p(lh_val) / (np.log1p(fsh_val) + 1e-5)

    # 3. Anthropometric & Metabolic Biomarkers
    if 'Weight (Kg)' in df_out.columns and 'Height(Cm)' in df_out.columns:
        height_m = df_out['Height(Cm)'].replace(0, 160) / 100.0
        df_out['Calculated_BMI'] = df_out['Weight (Kg)'] / (height_m ** 2)
        df_out['BMI_Overweight_Flag'] = (df_out['Calculated_BMI'] >= 25.0).astype(int)

    if 'Waist(inch)' in df_out.columns and 'Hip(inch)' in df_out.columns:
        hip_safe = df_out['Hip(inch)'].replace(0, 36)
        df_out['WHR_Calculated'] = df_out['Waist(inch)'] / hip_safe
        df_out['High_WHR_Flag'] = (df_out['WHR_Calculated'] >= 0.85).astype(int)

    # 4. Hyperandrogenism & Clinical Symptom Scores
    symptom_cols = [
        'Weight gain(Y/N)', 'hair growth(Y/N)', 'Skin darkening (Y/N)',
        'Hair loss(Y/N)', 'Pimples(Y/N)', 'Fast food (Y/N)'
    ]
    available_symptoms = [c for c in symptom_cols if c in df_out.columns]
    if available_symptoms:
        df_out['Metabolic_Symptom_Score'] = df_out[available_symptoms].sum(axis=1)
        
        if 'hair growth(Y/N)' in df_out.columns and 'Pimples(Y/N)' in df_out.columns:
            df_out['Hirsutism_Acne_Score'] = df_out['hair growth(Y/N)'] + df_out['Pimples(Y/N)']
            
        if 'Skin darkening (Y/N)' in df_out.columns and 'Weight gain(Y/N)' in df_out.columns:
            df_out['Skin_Metabolic_Score'] = df_out['Skin darkening (Y/N)'] + df_out['Weight gain(Y/N)']

    # 5. AMH Interactions
    if 'AMH(ng/mL)' in df_out.columns:
        amh_val = np.maximum(0, df_out['AMH(ng/mL)'])
        df_out['AMH_Log'] = np.log1p(amh_val)
        if 'Total_Follicles' in df_out.columns:
            df_out['AMH_Total_Follicles_Interaction'] = amh_val * df_out['Total_Follicles']

    # 6. Reserve Density & Menstrual Dynamics
    if 'Total_Follicles' in df_out.columns and 'Age (yrs)' in df_out.columns:
        df_out['Follicles_Per_Year'] = df_out['Total_Follicles'] / (df_out['Age (yrs)'] + 1e-5)

    if 'Cycle(R/I)' in df_out.columns and 'Cycle length(days)' in df_out.columns:
        df_out['Cycle_Severity_Index'] = df_out['Cycle(R/I)'] * df_out['Cycle length(days)']
        if 'Endometrium (mm)' in df_out.columns:
            df_out['Endometrium_Cycle_Ratio'] = df_out['Endometrium (mm)'] / (df_out['Cycle length(days)'] + 1e-5)

    added_count = df_out.shape[1] - df.shape[1]
    print(f"Domain Feature Engineering: Added {added_count} clinical interaction and Rotterdam consensus features.")
    return df_out
