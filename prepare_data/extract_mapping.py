"""
extract_mapping.py – Tìm hoặc tạo mapping drug index → tên, SE index → tên.

Chạy:
    python prepare_data/extract_mapping.py

Nếu pickle files chứa metadata (DataFrame/dict) → tự động trích xuất.
Nếu chỉ là numpy array → in hướng dẫn 3 phương án thay thế.
"""

import os
import pickle
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Datas")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_pkl(name: str):
    path = os.path.join(DATA_DIR, name)
    with open(path, "rb") as f:
        return pickle.load(f)


def try_extract_from_pickle():
    """Check if any pickle contains names as DataFrame index/columns or dict keys."""
    candidates = [
        "drug_side.pkl",
        "drug_mol.pkl",
        "drug_target.pkl",
        "glove_wordEmbedding.pkl",
        "side_effect_semantic.pkl",
    ]
    drug_names = None
    se_names = None

    for fname in candidates:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            continue

        obj = load_pkl(fname)

        try:
            import pandas as pd
            if isinstance(obj, pd.DataFrame):
                print(f"[{fname}] is DataFrame, shape={obj.shape}")
                print(f"  columns[:5] = {obj.columns[:5].tolist()}")
                print(f"  index[:5]   = {obj.index[:5].tolist()}")

                # drug_side.pkl: rows=drugs, cols=side_effects
                if fname == "drug_side.pkl":
                    idx = obj.index.tolist()
                    cols = obj.columns.tolist()
                    # Check if index contains drug names (not just integers)
                    if isinstance(idx[0], str):
                        drug_names = idx
                        print(f"  → Found drug names in index ({len(drug_names)})")
                    if isinstance(cols[0], str):
                        se_names = cols
                        print(f"  → Found SE names in columns ({len(se_names)})")
                continue
        except ImportError:
            pass

        if isinstance(obj, dict):
            keys = list(obj.keys())
            print(f"[{fname}] is dict, len={len(keys)}, keys[:5]={keys[:5]}")
            if isinstance(keys[0], str):
                if "drug" in fname.lower():
                    drug_names = keys
                    print(f"  → Possible drug names ({len(keys)})")
                elif "side" in fname.lower() or "glove" in fname.lower():
                    se_names = keys
                    print(f"  → Possible SE names ({len(keys)})")

    return drug_names, se_names


def save_mapping(names, output_path, id_col, name_col):
    """Save a list of names as CSV mapping file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    import csv
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", id_col, name_col])
        for i, name in enumerate(names):
            writer.writerow([i, "", name])
    print(f"Saved mapping to {output_path} ({len(names)} entries)")


def print_alternatives():
    """Print alternative ways to obtain mapping when pickle has no metadata."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║  KHÔNG TÌM THẤY MAPPING TRONG PICKLE FILES                 ║
║  Tất cả pickle files chỉ chứa numpy array (không có tên).  ║
║                                                              ║
║  BẠN CẦN LÀM 1 TRONG 3 CÁCH SAU:                          ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  (A) LIÊN HỆ TÁC GIẢ MSSF GỐC                             ║
║      Email: dingxlcse@csu.edu.cn                            ║
║      Yêu cầu: file mapping drug_index → DrugBank ID/name   ║
║               và    SE_index → ADReCS term/name             ║
║                                                              ║
║  (B) CROSS-REFERENCE VỚI DATASET GỐC                       ║
║      1. Tải SIDER database: http://sideeffects.embl.de/     ║
║      2. Tải DrugBank: https://go.drugbank.com/releases      ║
║      3. So sánh drug_side.pkl (757×994) với SIDER matrix    ║
║         → match pattern để xác định thứ tự drug/SE          ║
║                                                              ║
║  (C) REVERSE-LOOKUP TỪ MOL2VEC                              ║
║      1. Load drug_mol.pkl → (757, dim) molecular embeddings ║
║      2. Nếu đây là Mol2Vec → cần SMILES gốc để match       ║
║      3. Tải SMILES từ DrugBank/PubChem → encode Mol2Vec     ║
║      4. So sánh cosine similarity để match drug index        ║
║                                                              ║
║  SAU KHI CÓ MAPPING, tạo 2 file CSV:                       ║
║      data/drug_mapping.csv  (idx, drugbank_id, drug_name)   ║
║      data/se_mapping.csv    (idx, adrecs_id, se_name)       ║
║                                                              ║
║  ⚠ KHÔNG THỂ TIẾP TỤC BƯỚC 3 (text encoding) NẾU CHƯA     ║
║    CÓ MAPPING. Đây là BLOCKER.                              ║
╚══════════════════════════════════════════════════════════════╝
""")


def main():
    print("Attempting to extract drug/SE name mappings from pickle files...\n")

    drug_names, se_names = try_extract_from_pickle()

    drug_path = os.path.join(OUTPUT_DIR, "drug_mapping.csv")
    se_path = os.path.join(OUTPUT_DIR, "se_mapping.csv")

    found_any = False

    if drug_names:
        save_mapping(drug_names, drug_path, "drugbank_id", "drug_name")
        found_any = True
    else:
        print("\n❌ Drug names NOT found in pickle files.")

    if se_names:
        save_mapping(se_names, se_path, "adrecs_id", "se_name")
        found_any = True
    else:
        print("❌ SE names NOT found in pickle files.")

    if not found_any:
        print_alternatives()
    else:
        print(f"\n✅ Mapping files saved to {OUTPUT_DIR}/")
        print("Next step: Use these mappings in prepare_data/build_texts.py")

    # Also print shape of drug_side for confirmation
    fpath = os.path.join(DATA_DIR, "drug_side.pkl")
    if os.path.exists(fpath):
        obj = load_pkl("drug_side.pkl")
        arr = np.array(obj)
        print(f"\ndrug_side.pkl shape: {arr.shape}")
        print(f"  Unique values: {np.unique(arr)}")
        print(f"  Nonzero count: {np.count_nonzero(arr)}")
        print(f"  Expected: 757 drugs × 994 side effects")


if __name__ == "__main__":
    main()
