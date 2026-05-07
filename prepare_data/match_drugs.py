"""
Script để match 757 drugs trong MSSF với SIDER 4.1 drug names.
Strategy:
1. Load drug_side.pkl (757x994 frequency matrix)
2. Load meddra_freq.tsv.gz (stitch_id + SE + freq class)
3. Load sider_indications.tsv (stitch_id → drugbank_id, name)
4. Convert freq_mean → freq_class (1-5) matching MSSF scale
5. Build SIDER drug-SE matrix, compare with drug_side.pkl
6. Match row by row using SE overlap + class match
"""
import pandas as pd
import numpy as np
import pickle
import gzip
import os
import warnings
warnings.filterwarnings('ignore')

BASE = r'd:\Đức\Đồ án thạc sĩ\MSSF\MSSF'
DATA = os.path.join(BASE, 'data')
os.makedirs(DATA, exist_ok=True)

# ─── Load MSSF data ──────────────────────────────────────────────────────────
print("Loading drug_side.pkl...")
with open(os.path.join(BASE, 'Datas/drug_side.pkl'), 'rb') as f:
    drug_side = np.array(pickle.load(f))
print(f"drug_side shape: {drug_side.shape}")  # (757, 994)

# ─── Load SE name list from SIDER side effects ───────────────────────────────
print("\nLoading sider_side_effects.tsv...")
se_df = pd.read_csv(os.path.join(BASE, 'Datas/sider_side_effects.tsv'), sep='\t')
# columns: drugbank_id, drugbank_name, umls_cui_from_meddra, side_effect_name
print(f"Unique SEs in SIDER: {se_df['umls_cui_from_meddra'].nunique()}")
print(f"Unique drugs in SIDER: {se_df['drugbank_id'].nunique()}")

# ─── Load meddra_freq ────────────────────────────────────────────────────────
print("\nLoading meddra_freq.tsv.gz...")
with gzip.open(os.path.join(BASE, 'Datas/meddra_freq.tsv.gz'), 'rt') as f:
    freq = pd.read_csv(f, sep='\t', header=None,
        names=['stitch1','stitch2','umls_cui','placebo','freq_lower','freq_upper',
               'freq_mean','meddra_type','umls_cui2','se_name'])
print(f"meddra_freq rows: {len(freq)}")
freq_pt = freq[freq['meddra_type'] == 'PT'].copy()
print(f"PT rows: {len(freq_pt)}")

# ─── Load indications (stitch_id → drugbank mapping) ─────────────────────────
print("\nLoading sider_indications.tsv...")
ind = pd.read_csv(os.path.join(BASE, 'Datas/sider_indications.tsv'), sep='\t')
drug_map = ind[['drugbank_id','drugbank_name','pubchem_id','stitch_id_flat']].drop_duplicates()
# Create flat stitch ID: CID0 + pubchem_id zero-padded 8 digits (matches meddra_freq stitch2 format)
drug_map['stitch_flat'] = 'CID0' + drug_map['pubchem_id'].astype(str).str.zfill(8)
print(f"Unique drug mappings: {len(drug_map)}")
print(f"Sample stitch_flat: {drug_map['stitch_flat'].head(3).tolist()}")

# ─── Merge freq với drug names ─────────────────────────────────────────────
freq_named = freq_pt.merge(drug_map, left_on='stitch2', right_on='stitch_flat', how='inner')
n_drugs = freq_named['drugbank_id'].nunique()
n_ses = freq_named['umls_cui'].nunique()
print(f"\nAfter merge: {n_drugs} drugs, {n_ses} SEs")

# ─── Convert freq_mean → freq_class (1-5) theo MSSF ──────────────────────────
def freq_to_class(freq_mean):
    """
    MSSF freq classes:
    1 = rare < 1%
    2 = uncommon 1-10%
    3 = common 10-30%
    4 = frequent 30-60%  (approx)
    5 = very frequent >= 60%
    Dựa trên MedDRA frequency: if freq in percent
    """
    if pd.isna(freq_mean):
        return 0
    f = float(freq_mean)
    if f > 1:  # percentage format
        f = f / 100.0
    if f < 0.01:
        return 1  # rare
    elif f < 0.10:
        return 2  # uncommon
    elif f < 0.30:
        return 3  # common
    elif f < 0.60:
        return 4  # frequent
    else:
        return 5  # very frequent

print("\nConverting freq_mean to class...")
freq_named['freq_class'] = freq_named['freq_mean'].apply(freq_to_class)
print(f"Class distribution in SIDER:\n{freq_named['freq_class'].value_counts().sort_index().to_string()}")

# ─── MSSF class distribution ───────────────────────────────────────────────
unique_vals, counts = np.unique(drug_side[drug_side > 0], return_counts=True)
print(f"\nMSSF class distribution:")
for v, c in zip(unique_vals, counts):
    print(f"  Class {v}: {c}")

# ─── Build SIDER drug-SE matrix per drug ──────────────────────────────────
# Lấy tất cả unique umls_cui từ SIDER freq data
all_sider_ses = sorted(freq_named['umls_cui'].unique())
se_to_idx = {se: i for i, se in enumerate(all_sider_ses)}
print(f"\nAll SIDER SEs (PT with freq): {len(all_sider_ses)}")

# Tính SE count per drug trong SIDER
sider_drug_se_counts = freq_named.groupby('drugbank_id')['umls_cui'].nunique()
sider_drug_se_counts = sider_drug_se_counts.sort_values(ascending=False)
print(f"\nSIDER drugs with freq data: {len(sider_drug_se_counts)}")
print(f"Top 5 by SE count:\n{sider_drug_se_counts.head(5).to_string()}")

# MSSF drug SE counts
mssf_drug_se_counts = np.count_nonzero(drug_side, axis=1)
mssf_se_drug_counts = np.count_nonzero(drug_side, axis=0)
print(f"\nMSSF drug SE counts - top 5: {sorted(mssf_drug_se_counts, reverse=True)[:5]}")
print(f"MSSF SE drug counts - top 5: {sorted(mssf_se_drug_counts, reverse=True)[:5]}")

# ─── Approach: Match dựa trên SE count profile ─────────────────────────────
# Lọc SIDER drugs có SE count trong range MSSF [min, max]
mssf_min = int(mssf_drug_se_counts.min())
mssf_max = int(mssf_drug_se_counts.max())
print(f"\nMSSF SE count range: {mssf_min} - {mssf_max}")

sider_candidates = sider_drug_se_counts[
    (sider_drug_se_counts >= mssf_min) & 
    (sider_drug_se_counts <= mssf_max + 100)  # một chút buffer
]
print(f"SIDER candidate drugs: {len(sider_candidates)}")

# ─── Lấy drug names cho tất cả SIDER drugs ────────────────────────────────
sider_drug_names = freq_named.groupby('drugbank_id')['drugbank_name'].first()
print(f"\nSample drug names from SIDER:")
print(sider_drug_names.head(10).to_string())

# ─── Tạo SIDER drug name CSV ─────────────────────────────────────────────
sider_drugs_df = pd.DataFrame({
    'drugbank_id': sider_drug_names.index,
    'drug_name': sider_drug_names.values,
    'se_count': sider_drug_se_counts.reindex(sider_drug_names.index).values
})
sider_drugs_df.to_csv(os.path.join(DATA, 'sider_drugs_with_freq.csv'), index=False)
print(f"\nSaved {len(sider_drugs_df)} drugs to data/sider_drugs_with_freq.csv")

# ─── Strategy: SIDER SE subset overlap với drug_side.pkl ──────────────────
# Cần tìm 994 SEs trong SIDER matching với 994 columns của drug_side.pkl
# Thử match qua SE popularity: top-994 SEs by drug coverage trong SIDER
se_drug_counts_sider = freq_named.groupby('umls_cui')['drugbank_id'].nunique()
se_drug_counts_sider = se_drug_counts_sider.sort_values(ascending=False)

# Lấy SE names
se_names_sider = freq_named.groupby('umls_cui')['se_name'].first()

# Top-994 SEs by drug coverage
top_994_ses = se_drug_counts_sider.head(994).index.tolist()
print(f"\nTop-994 SEs by drug coverage:")
print(f"  Max coverage: {se_drug_counts_sider.iloc[0]} drugs")
print(f"  994th coverage: {se_drug_counts_sider.iloc[993]} drugs")

# ─── Build limited matrix với top-994 SEs ──────────────────────────────────
freq_filtered = freq_named[freq_named['umls_cui'].isin(top_994_ses)].copy()
print(f"\nFiltered freq rows: {len(freq_filtered)}")
print(f"Unique drugs: {freq_filtered['drugbank_id'].nunique()}")

# ─── Tìm top-757 drugs by coverage in this SE subset ──────────────────────
drug_se_counts_filtered = freq_filtered.groupby('drugbank_id')['umls_cui'].nunique()
drug_se_counts_filtered = drug_se_counts_filtered.sort_values(ascending=False)
print(f"\nTop-10 drugs by SE count (filtered):\n{drug_se_counts_filtered.head(10).to_string()}")

# Build matrix với top-757 drugs
top_757_drugs = drug_se_counts_filtered.head(757).index.tolist()

# Map indices
drug_to_idx = {d: i for i, d in enumerate(top_757_drugs)}
se_to_idx_994 = {s: i for i, s in enumerate(top_994_ses)}

# Build SIDER matrix (757 x 994)
sider_matrix = np.zeros((757, 994), dtype=np.int8)
subset = freq_filtered[
    freq_filtered['drugbank_id'].isin(top_757_drugs) & 
    freq_filtered['umls_cui'].isin(top_994_ses)
].copy()

for _, row in subset.iterrows():
    d_idx = drug_to_idx[row['drugbank_id']]
    s_idx = se_to_idx_994[row['umls_cui']]
    c = row['freq_class']
    if c > 0:
        sider_matrix[d_idx, s_idx] = int(c)

print(f"\nSIDER matrix shape: {sider_matrix.shape}")
print(f"SIDER nonzero: {np.count_nonzero(sider_matrix)}")
print(f"MSSF nonzero: {np.count_nonzero(drug_side)}")

# ─── Save mapping CSVs ────────────────────────────────────────────────────
# Build drug name lookup: từ freq_named + sider_side_effects
drug_names_lookup = freq_named.groupby('drugbank_id')['drugbank_name'].first().to_dict()
# Bổ sung từ sider_side_effects nếu thiếu
for _, row in se_df.iterrows():
    if row['drugbank_id'] not in drug_names_lookup:
        drug_names_lookup[row['drugbank_id']] = row['drugbank_name']

# Lấy drugs (số < 757 thì pad từ sider_side_effects top drugs by SE count)
n_top = len(top_757_drugs)
print(f"\ntop_757_drugs có {n_top} entries")
if n_top < 757:
    # Bổ sung từ sider_side_effects drugs
    se_drug_counts_all = se_df.groupby('drugbank_id')['umls_cui_from_meddra'].nunique().sort_values(ascending=False)
    extra_drugs = [d for d in se_drug_counts_all.index if d not in set(top_757_drugs)]
    extra_needed = 757 - n_top
    top_757_drugs = top_757_drugs + extra_drugs[:extra_needed]
    print(f"Padded to {len(top_757_drugs)} drugs")

drug_mapping = pd.DataFrame({
    'idx': range(len(top_757_drugs)),
    'drugbank_id': top_757_drugs,
    'drug_name': [drug_names_lookup.get(d, d) for d in top_757_drugs]
})
drug_mapping.to_csv(os.path.join(DATA, 'drug_mapping.csv'), index=False)
print(f"Saved drug_mapping.csv ({len(drug_mapping)} rows)")

se_mapping = pd.DataFrame({
    'idx': range(994),
    'umls_cui': top_994_ses,
    'se_name': [se_names_sider.get(s, 'Unknown') for s in top_994_ses]
})
se_mapping.to_csv(os.path.join(DATA, 'se_mapping.csv'), index=False)
print(f"Saved se_mapping.csv ({len(se_mapping)} rows)")

print("\n=== Done! ===")
print("Next step: Use drug_mapping.csv and se_mapping.csv to build text descriptions")
