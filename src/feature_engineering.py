import pandas as pd
import numpy as np
from typing import Tuple, List

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names by stripping whitespace and unifying common variants."""
    df_clean = df.copy()
    df_clean.columns = [c.strip() for c in df_clean.columns]
    
    # Standardize frequent space differences in clinical labels
    rename_dict = {}
    for c in df_clean.columns:
        c_lower = c.lower().replace(" ", "").replace("_", "")
        if "age" in c_lower and "yr" in c_lower:
            rename_dict[c] = "Age (yrs)"
        elif "weight" in c_lower and "kg" in c_lower:
            rename_dict[c] = "Weight (Kg)"
        elif "height" in c_lower and "cm" in c_lower:
            rename_dict[c] = "Height(Cm)"
        elif "follicleno.(l)" in c_lower or "follicleno(l)" in c_lower:
            rename_dict[c] = "Follicle No. (L)"
        elif "follicleno.(r)" in c_lower or "follicleno(r)" in c_lower:
            rename_dict[c] = "Follicle No. (R)"
        elif "amh" in c_lower and "ng/ml" in c_lower:
            rename_dict[c] = "AMH(ng/mL)"
        elif "fsh" in c_lower and "lh" not in c_lower and "miu/ml" in c_lower:
            rename_dict[c] = "FSH(mIU/mL)"
        elif "lh" in c_lower and "fsh" not in c_lower and "miu/ml" in c_lower:
            rename_dict[c] = "LH(mIU/mL)"
        elif "waist:hip" in c_lower or "whr" in c_lower:
            rename_dict[c] = "Waist:Hip Ratio"
        elif "endometrium" in c_lower:
            rename_dict[c] = "Endometrium (mm)"
        elif "cycle(r/i)" in c_lower:
            rename_dict[c] = "Cycle(R/I)"
        elif "cyclelength" in c_lower:
            rename_dict[c] = "Cycle length(days)"

    if rename_dict:
        df_clean = df_clean.rename(columns=rename_dict)
    return df_clean


def engineer_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers novel domain-informed endocrinological, ultrasound, and metabolic features
    aligned with international Rotterdam consensus diagnostic criteria:
    
    1. Ultrasound & Morphological Markers (Rotterdam PCOM):
       - 1. Total Follicle Count = Follicle (L) + Follicle (R)
       - 2. Follicle Difference = |Follicle (L) - Follicle (R)|
       - 3. Follicle Asymmetry Index = |L - R| / (Total Follicles + 1e-5)
       - 4. Geometric Mean Follicle Count = sqrt(max(0, L) * max(0, R))
       - 5. Rotterdam PCOM Binary Flag: Follicle Count >= 10 in either ovary (Rotterdam 2003)
       - 6. Severe PCOM Flag: Follicle Count >= 12 in either ovary (AE-PCOS criteria)
       - 7. Mean Follicle Size = (Avg. F size L + Avg. F size R) / 2
       - 8. Follicle Size Disparity = |Avg. F size L - Avg. F size R|
       
    2. Endocrine & Hormonal Biomarkers:
       - 9. LH / FSH Ratio Calculated = LH / (FSH + 1e-5)
       - 10. High LH/FSH Hallmark Flag (Ratio > 2.0)
       - 11. LH/FSH Log Ratio = log1p(LH) / (log1p(FSH) + 1e-5)
       - 12. AMH Log Transform = log1p(AMH)
       - 13. AMH * Total Follicles Interaction (Granulosa cell hypersecretion proxy)
       - 14. Hormonal-Metabolic Composite = (AMH * LH/FSH) / (FSH + 1e-5)
       
    3. Anthropometric & Metabolic Indicators:
       - 15. Calculated BMI = Weight (kg) / (Height (m))^2
       - 16. BMI Overweight Flag (BMI >= 25.0)
       - 17. Waist-to-Hip Ratio (WHR) = Waist / Hip
       - 18. Android Central Adiposity Flag (WHR >= 0.85)
       - 19. Mean Arterial Pressure (MAP) = (Systolic BP + 2 * Diastolic BP) / 3
       
    4. Hyperandrogenism & Clinical Clustering:
       - 20. Total Metabolic Symptom Score (sum of all 6 clinical burden flags)
       - 21. Hirsutism & Acne Triad Score = hair growth + pimples + skin darkening
       - 22. Skin Acanthosis & Insulin Resistance Index = skin darkening + weight gain
       - 23. Follicles Per Year of Age (Ovarian Reserve Density)
       - 24. Endometrium to Cycle Length Dynamic Ratio
       - 25. Cycle Severity Burden Index = Cycle(R/I) * Cycle length(days)
    """
    df_out = normalize_column_names(df)

    # 1. Ultrasound & Follicle Metrics
    if 'Follicle No. (L)' in df_out.columns and 'Follicle No. (R)' in df_out.columns:
        f_l = pd.to_numeric(df_out['Follicle No. (L)'], errors='coerce').fillna(0)
        f_r = pd.to_numeric(df_out['Follicle No. (R)'], errors='coerce').fillna(0)
        
        # 1-6 Follicle Metrics
        df_out['Total_Follicles'] = f_l + f_r
        df_out['Follicle_Diff'] = (f_l - f_r).abs()
        df_out['Follicle_Asymmetry_Index'] = df_out['Follicle_Diff'] / (df_out['Total_Follicles'] + 1e-5)
        df_out['Follicle_Geometric_Mean'] = np.sqrt(np.maximum(0, f_l) * np.maximum(0, f_r))
        df_out['PCOM_Rotterdam_Flag'] = ((f_l >= 10) | (f_r >= 10)).astype(int)
        df_out['PCOM_Severe_Flag'] = ((f_l >= 12) | (f_r >= 12)).astype(int)
        
        # 7-8 Follicle Size Metrics
        f_size_l = [c for c in df_out.columns if 'avg' in c.lower() and '(l)' in c.lower()]
        f_size_r = [c for c in df_out.columns if 'avg' in c.lower() and '(r)' in c.lower()]
        if f_size_l and f_size_r:
            sz_l = pd.to_numeric(df_out[f_size_l[0]], errors='coerce').fillna(0)
            sz_r = pd.to_numeric(df_out[f_size_r[0]], errors='coerce').fillna(0)
            df_out['Mean_Follicle_Size_Both'] = (sz_l + sz_r) / 2.0
            df_out['Follicle_Size_Disparity'] = (sz_l - sz_r).abs()

    # 2. Hormonal & Endocrine Indicators
    if 'LH(mIU/mL)' in df_out.columns and 'FSH(mIU/mL)' in df_out.columns:
        lh_val = pd.to_numeric(df_out['LH(mIU/mL)'], errors='coerce').fillna(0)
        fsh_val = pd.to_numeric(df_out['FSH(mIU/mL)'], errors='coerce').fillna(0)
        fsh_safe = fsh_val.replace(0, 0.01)
        
        # 9-11 LH/FSH Metrics
        df_out['LH_FSH_Ratio_Calculated'] = lh_val / fsh_safe
        df_out['High_LH_FSH_Flag'] = (df_out['LH_FSH_Ratio_Calculated'] > 2.0).astype(int)
        df_out['LH_FSH_Log_Ratio'] = np.log1p(np.maximum(0, lh_val)) / (np.log1p(np.maximum(0, fsh_val)) + 1e-5)

    # 12-14 AMH Biomarkers & Composite
    if 'AMH(ng/mL)' in df_out.columns:
        amh_val = pd.to_numeric(df_out['AMH(ng/mL)'], errors='coerce').fillna(0)
        amh_nonneg = np.maximum(0, amh_val)
        df_out['AMH_Log'] = np.log1p(amh_nonneg)
        if 'Total_Follicles' in df_out.columns:
            df_out['AMH_Total_Follicles_Interaction'] = amh_nonneg * df_out['Total_Follicles']
        if 'LH_FSH_Ratio_Calculated' in df_out.columns and 'FSH(mIU/mL)' in df_out.columns:
            fsh_safe = pd.to_numeric(df_out['FSH(mIU/mL)'], errors='coerce').fillna(0).replace(0, 0.01)
            df_out['Hormonal_Metabolic_Composite'] = (amh_nonneg * df_out['LH_FSH_Ratio_Calculated']) / (fsh_safe + 1e-5)

    # 3. Anthropometric & Metabolic Biomarkers
    if 'Weight (Kg)' in df_out.columns and 'Height(Cm)' in df_out.columns:
        wt = pd.to_numeric(df_out['Weight (Kg)'], errors='coerce').fillna(60)
        ht = pd.to_numeric(df_out['Height(Cm)'], errors='coerce').fillna(160)
        ht_m = ht.replace(0, 160) / 100.0
        # 15-16 BMI metrics
        df_out['Calculated_BMI'] = wt / (ht_m ** 2)
        df_out['BMI_Overweight_Flag'] = (df_out['Calculated_BMI'] >= 25.0).astype(int)

    if 'Waist(inch)' in df_out.columns and 'Hip(inch)' in df_out.columns:
        waist = pd.to_numeric(df_out['Waist(inch)'], errors='coerce').fillna(32)
        hip = pd.to_numeric(df_out['Hip(inch)'], errors='coerce').fillna(38).replace(0, 38)
        # 17-18 WHR
        df_out['WHR_Calculated'] = waist / hip
        df_out['High_WHR_Flag'] = (df_out['WHR_Calculated'] >= 0.85).astype(int)

    # 19. Blood Pressure & Mean Arterial Pressure (MAP)
    bp_sys_cols = [c for c in df_out.columns if 'systolic' in c.lower()]
    bp_dia_cols = [c for c in df_out.columns if 'diastolic' in c.lower()]
    if bp_sys_cols and bp_dia_cols:
        sbp = pd.to_numeric(df_out[bp_sys_cols[0]], errors='coerce').fillna(120)
        dbp = pd.to_numeric(df_out[bp_dia_cols[0]], errors='coerce').fillna(80)
        df_out['MAP_Blood_Pressure'] = (sbp + 2.0 * dbp) / 3.0

    # 4. Hyperandrogenism & Clinical Symptom Scores
    symptom_cols = [
        'Weight gain(Y/N)', 'hair growth(Y/N)', 'Skin darkening (Y/N)',
        'Hair loss(Y/N)', 'Pimples(Y/N)', 'Fast food (Y/N)'
    ]
    available_symptoms = [c for c in symptom_cols if c in df_out.columns]
    if available_symptoms:
        # Convert symptoms to 0/1 numeric
        for s in available_symptoms:
            df_out[s] = pd.to_numeric(df_out[s], errors='coerce').fillna(0).astype(int)
        
        # 20. Total burden
        df_out['Metabolic_Symptom_Score'] = df_out[available_symptoms].sum(axis=1)
        
        # 21-22 Triads
        if 'hair growth(Y/N)' in df_out.columns and 'Pimples(Y/N)' in df_out.columns and 'Skin darkening (Y/N)' in df_out.columns:
            df_out['Hirsutism_Acne_Score'] = df_out['hair growth(Y/N)'] + df_out['Pimples(Y/N)'] + df_out['Skin darkening (Y/N)']
        elif 'hair growth(Y/N)' in df_out.columns and 'Pimples(Y/N)' in df_out.columns:
            df_out['Hirsutism_Acne_Score'] = df_out['hair growth(Y/N)'] + df_out['Pimples(Y/N)']
            
        if 'Skin darkening (Y/N)' in df_out.columns and 'Weight gain(Y/N)' in df_out.columns:
            df_out['Skin_Metabolic_Score'] = df_out['Skin darkening (Y/N)'] + df_out['Weight gain(Y/N)']

    # 5. Reserve Density & Menstrual Dynamics
    if 'Total_Follicles' in df_out.columns and 'Age (yrs)' in df_out.columns:
        age_val = pd.to_numeric(df_out['Age (yrs)'], errors='coerce').fillna(28)
        # 23. Reserve density
        df_out['Follicles_Per_Year'] = df_out['Total_Follicles'] / (age_val + 1e-5)

    if 'Cycle(R/I)' in df_out.columns and 'Cycle length(days)' in df_out.columns:
        cycle_ri = pd.to_numeric(df_out['Cycle(R/I)'], errors='coerce').fillna(2)
        cycle_len = pd.to_numeric(df_out['Cycle length(days)'], errors='coerce').fillna(5)
        # 24-25 Dynamic ratios
        df_out['Cycle_Severity_Index'] = cycle_ri * cycle_len
        if 'Endometrium (mm)' in df_out.columns:
            endo = pd.to_numeric(df_out['Endometrium (mm)'], errors='coerce').fillna(7)
            df_out['Endometrium_Cycle_Ratio'] = endo / (cycle_len + 1e-5)

    added_count = df_out.shape[1] - df.shape[1]
    print(f"Domain Feature Engineering: Added {added_count} clinical interaction and Rotterdam consensus features.")
    return df_out

