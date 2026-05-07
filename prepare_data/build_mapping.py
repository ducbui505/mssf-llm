"""
build_mapping.py – Cross-reference SIDER 4.1 với drug_side.pkl của MSSF
để xác định drug index → tên thuốc, SE index → tên tác dụng phụ.

Chạy:
    python prepare_data/build_mapping.py

Output:
    data/drug_mapping.csv  (idx, drugbank_id, drug_name)
    data/se_mapping.csv    (idx, umls_cui, se_name)
"""

import os
import gzip
import pickle
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATAS = os.path.join(BASE, "Datas")
OUTPUT = os.path.join(BASE, "data")
os.makedirs(OUTPUT, exist_ok=True)

# ──────────────────────────────────────────────
# 1. Load drug_side.pkl (757 × 994)
# ──────────────────────────────────────────────
print("Loading drug_side.pkl ...")
with open(os.path.join(DATAS, "drug_side.pkl"), "rb") as f:
    drug_side = np.array(pickle.load(f))
print(f"  shape: {drug_side.shape}, nonzero: {np.count_nonzero(drug_side)}")
n_drugs, n_se = drug_side.shape   # 757, 994

# ──────────────────────────────────────────────
# 2. Load meddra_freq.tsv.gz (SIDER frequency data)
# Columns: stitch_id1, stitch_id2, umls_cui_from_meddra, placebo,
#          freq_lower, freq_upper, freq_mean, meddra_type,
#          umls_cui_from_meddra2, se_name
# ──────────────────────────────────────────────
print("\nLoading meddra_freq.tsv.gz ...")
freq_path = os.path.join(DATAS, "meddra_freq.tsv.gz")
with gzip.open(freq_path, "rt", encoding="utf-8") as f:
    freq_df = pd.read_csv(f, sep="\t", header=None,
        names=["stitch1","stitch2","umls_cui","placebo",
               "freq_lower","freq_upper","freq_mean",
               "meddra_type","umls_cui2","se_name"])

print(f"  rows: {len(freq_df)}")
print(f"  columns: {freq_df.columns.tolist()}")
print(f"  sample:\n{freq_df.head(3).to_string()}")

# ──────────────────────────────────────────────
# 3. Load side-effects.tsv (all drug-SE pairs with names)
# Columns: drugbank_id, drugbank_name, umls_cui_from_meddra, side_effect_name
# ──────────────────────────────────────────────
print("\nLoading sider_side_effects.tsv ...")
se_path = os.path.join(DATAS, "sider_side_effects.tsv")
se_df = pd.read_csv(se_path, sep="\t")
print(f"  rows: {len(se_df)}")
print(f"  columns: {se_df.columns.tolist()}")
print(f"  unique drugs: {se_df['drugbank_id'].nunique()}")
print(f"  unique SEs: {se_df['umls_cui_from_meddra'].nunique()}")

# ──────────────────────────────────────────────
# 4. Xây dựng frequency matrix từ SIDER
#    Chuyển frequency về 5 class như MSSF (1–5)
#    MSSF dùng: 1=rare, 2=infrequent, 3=common, 4=frequent, 5=very frequent
# ──────────────────────────────────────────────
print("\nBuilding frequency class from SIDER freq data ...")

def parse_freq(val):
    """Parse giá trị frequency: '21%' → 0.21, '0.21' → 0.21, NaN → NaN."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace('%', '')
    try:
        v = float(s)
        return v / 100 if v > 1 else v  # nếu >1 thì đang ở dạng % rồi
    except:
        return np.nan

freq_df["lower_f"] = freq_df["freq_lower"].apply(parse_freq)
freq_df["upper_f"] = freq_df["freq_upper"].apply(parse_freq)
freq_df["mean_f"]  = freq_df["freq_mean"].apply(parse_freq)

def freq_to_class(lower, upper, mean):
    """Map frequency → class 1-5 như MSSF paper."""
    val = mean if not np.isnan(mean) else \
          (lower + upper) / 2 if not np.isnan(lower) and not np.isnan(upper) else \
          lower if not np.isnan(lower) else \
          upper if not np.isnan(upper) else np.nan
    if np.isnan(val): return 0
    if val < 0.001:   return 1  # rare < 0.1%
    elif val < 0.01:  return 2  # infrequent 0.1%–1%
    elif val < 0.1:   return 3  # common 1%–10%
    elif val < 0.5:   return 4  # frequent 10%–50%
    else:             return 5  # very frequent ≥50%

freq_df["freq_class"] = freq_df.apply(
    lambda r: freq_to_class(r["lower_f"], r["upper_f"], r["mean_f"]), axis=1)

# Kiểm tra unique placebo values
print(f"  placebo unique values: {freq_df['placebo'].dropna().unique()[:5].tolist()}")

# Chỉ giữ non-placebo và có freq data
freq_clinical = freq_df[freq_df["freq_class"] > 0].copy()
print(f"  Clinical freq pairs (class>0): {len(freq_clinical)}")

# ──────────────────────────────────────────────
# 5. Cross-reference: tìm 757 drugs và 994 SEs
#    Lấy số lượng SE có freq data cho mỗi drug
# ──────────────────────────────────────────────
print("\nCross-referencing to find 757 drugs and 994 SEs ...")

# Merge với se_df để có drugbank_id và tên
# stitch1 trong meddra_freq là PubChem CID (dạng CID1xxxxxx)
# Cần map stitch → drugbank_id qua se_df

# Cách đơn giản: dùng se_df để lấy unique drugs và SEs
# rồi tìm subset mà khi tạo ma trận freq, số drug=757, số SE=994

# Lấy tất cả SEs có tên
all_se_names = se_df.groupby("umls_cui_from_meddra")["side_effect_name"].first().reset_index()
all_se_names.columns = ["umls_cui", "se_name"]
print(f"  Total unique SEs in SIDER: {len(all_se_names)}")

all_drug_names = se_df.groupby("drugbank_id")["drugbank_name"].first().reset_index()
print(f"  Total unique drugs in SIDER: {len(all_drug_names)}")

# ──────────────────────────────────────────────
# 6. Tìm subset khớp với drug_side.pkl
#    Đếm số SE nonzero cho mỗi drug trong drug_side.pkl
#    Tìm drugs trong SIDER có cùng pattern
# ──────────────────────────────────────────────
drug_se_counts = np.count_nonzero(drug_side, axis=1)   # (757,) - số SE có freq mỗi drug
se_drug_counts = np.count_nonzero(drug_side, axis=0)    # (994,) - số drug có freq mỗi SE

print(f"\n  drug_side stats:")
print(f"    Drug SE counts: min={drug_se_counts.min()}, max={drug_se_counts.max()}, mean={drug_se_counts.mean():.1f}")
print(f"    SE drug counts: min={se_drug_counts.min()}, max={se_drug_counts.max()}, mean={se_drug_counts.mean():.1f}")
print(f"    Total nonzero: {np.count_nonzero(drug_side)}")

# Phân phối giá trị trong drug_side (các class 1-5)
for v in range(1, 6):
    cnt = np.sum(drug_side == v)
    print(f"    Class {v}: {cnt} pairs")

# ──────────────────────────────────────────────
# 7. Output: In thông tin để quyết định approach tiếp theo
# ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"SIDER has {all_drug_names.shape[0]} drugs, {all_se_names.shape[0]} unique SEs")
print(f"MSSF needs {n_drugs} drugs, {n_se} SEs")
print(f"\nApproach: Cần liên hệ tác giả MSSF hoặc match bằng frequency pattern")
print(f"\nFirst 10 SIDER drugs:")
print(all_drug_names.head(10).to_string())
print(f"\nFirst 10 SIDER SEs:")
print(all_se_names.head(10).to_string())

# Save raw SIDER data để dùng sau
all_drug_names.to_csv(os.path.join(OUTPUT, "sider_all_drugs.csv"), index=False)
all_se_names.to_csv(os.path.join(OUTPUT, "sider_all_ses.csv"), index=False)
print(f"\nSaved:")
print(f"  {os.path.join(OUTPUT, 'sider_all_drugs.csv')} ({len(all_drug_names)} drugs)")
print(f"  {os.path.join(OUTPUT, 'sider_all_ses.csv')} ({len(all_se_names)} SEs)")
