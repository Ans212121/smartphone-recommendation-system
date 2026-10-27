# Group 6: Smartphone Recommendation System — Real Dataset Version

## Group Members

- Shadi Dhaifallah Al-Zaidi
- Mohammed Majed Al-Bakhit
- Anas Ibrahim Al-Mutairi
- Abdulaziz Zayed Al-Mutairi


**Dataset:** Kaggle Global Mobile Phone Specifications & Prices (2026), uploaded archive.zip. 6,620 source rows; actual qualifying smartphone rows are listed in results/real_results.json.

**Currency:** BDT (Bangladeshi taka), **NOT** USD or SAR. Prices reflect publisher listings, not live offers.

**Features:** numerical RAM, storage, battery, main camera, display size, and rating (original 0–10 rescaled to 0–5). Extracted numbers may need manual verification on compound text fields. Missing features are median-imputed; populated outliers are excluded using documented plausibility ranges. Smartphones/Mobiles were selected during source preparation; runtime screens implausible feature-phone-like specs. The bundled CSV is preserved unmodified.

Install: `pip install -r requirements.txt`
Run GUI: `streamlit run app.py`
Run CLI: `python run_cli.py --budget 30000 --priority gaming --method hybrid`

Methods: content cosine similarity, utility weighted ranking, hybrid weighted blending. Validation covers budget compliance and descriptive comparisons; there is no labeled preference ground truth for precision/recall.

Dataset provenance: https://www.kaggle.com/datasets/utsh0dey/global-mobile-phone-specifications-and-prices


## Quality safeguards (revised October 2026)

The original processed CSV remains unchanged. The recommender filters RAM 2–32 GB, storage 16–2048 GB, battery 2500–20000 mAh, camera 2–250 MP, screen 5–9 inches, and rating 0–5 when populated; missing features are imputed. Thresholds are pragmatic, not guarantees of source accuracy. Gaming weights RAM and storage but does **not** benchmark CPU/GPU. No claims of precision/recall or recommendation accuracy are made. Results and slides were regenerated after filtering.

Run verification: `python -m pytest -q` (pytest required for tests).
