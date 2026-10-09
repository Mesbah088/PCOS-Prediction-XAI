import os
import urllib.request
import ssl

save_dir = r"c:\Users\user\OneDrive\Desktop\AI_ML\PCOS-Prediction-XAI\papers"
os.makedirs(save_dir, exist_ok=True)

papers = [
    {
        "filename": "05_Frontiers_Explainable_ML_Nomogram_PCOS_2025.pdf",
        "url": "https://www.frontiersin.org/articles/10.3389/fendo.2025.1719631/pdf",
        "title": "Frontiers in Endocrinology (Q1, 2025) - Explainable ML and Nomogram for PCOS"
    },
    {
        "filename": "09_Nature_ScientificReports_PCOS_ML_2022_Q1.pdf",
        "url": "https://www.nature.com/articles/s41598-022-21724-0.pdf",
        "title": "Nature Scientific Reports (Q1, 2022) - Extended ML for PCOS Detection"
    },
    {
        "filename": "10_PLOS_ONE_FNet_PCOS_DeepLearning_2024_Q1.pdf",
        "url": "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0307571&type=printable",
        "title": "PLOS ONE (Q1, 2024) - F-Net Follicles Net for PCOS Diagnosis"
    },
    {
        "filename": "11_PLOS_ONE_PCOS_Biomarkers_Ensemble_ML_2024_Q1.pdf",
        "url": "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0315418&type=printable",
        "title": "PLOS ONE (Q1, 2024) - Screening of Serum Biomarkers via Ensemble ML"
    }
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

print(f"Starting downloads into: {save_dir}\n")

for p in papers:
    filepath = os.path.join(save_dir, p["filename"])
    print(f"Fetching: {p['title']}...")
    try:
        req = urllib.request.Request(p["url"], headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp, open(filepath, "wb") as out_f:
            out_f.write(resp.read())
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  --> Saved: {p['filename']} ({size_kb:.1f} KB)\n")
    except Exception as e:
        print(f"  --> Error downloading {p['filename']}: {e}\n")

print("Done downloading papers.")
