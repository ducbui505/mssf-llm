"""
inspect_pickle.py – Khảo sát metadata của tất cả pickle files trong Datas/.

Chạy:
    python prepare_data/inspect_pickle.py

Output: In ra type, shape, dtype, sample values của mỗi file.
Nếu file là DataFrame/dict → in thêm columns/keys → dùng cho mapping.
"""

import os
import pickle
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Datas")

FILES = [
    "drug_side.pkl",
    "drug_mol.pkl",
    "glove_wordEmbedding.pkl",
    "side_effect_semantic.pkl",
    "Text_similarity_one.pkl",
    "Text_similarity_two.pkl",
    "Text_similarity_three.pkl",
    "Text_similarity_four.pkl",
    "Text_similarity_five.pkl",
    "fingerprint_similarity.pkl",
    "drug_target.pkl",
    "drug_pathway_similarity.pkl",
    "drug_pathway_enzyme_similarity.pkl",
]


def inspect_file(filepath: str) -> None:
    name = os.path.basename(filepath)
    print(f"\n{'=' * 60}")
    print(f"FILE: {name}")
    print(f"{'=' * 60}")

    with open(filepath, "rb") as f:
        obj = pickle.load(f)

    print(f"  type : {type(obj).__module__}.{type(obj).__name__}")

    # --- pandas DataFrame ---
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            print(f"  shape  : {obj.shape}")
            print(f"  columns: {obj.columns.tolist()}")
            print(f"  index  : {obj.index[:5].tolist()} ...")
            print(f"  dtypes :\n{obj.dtypes}")
            print(f"  head(3):\n{obj.head(3)}")
            return
    except ImportError:
        pass

    # --- dict ---
    if isinstance(obj, dict):
        print(f"  len(keys): {len(obj)}")
        keys_sample = list(obj.keys())[:10]
        print(f"  keys[:10]: {keys_sample}")
        for k in keys_sample[:3]:
            v = obj[k]
            vtype = type(v).__name__
            vshape = getattr(v, "shape", None)
            print(f"    [{k}] type={vtype}, shape={vshape}")
        return

    # --- list / tuple ---
    if isinstance(obj, (list, tuple)):
        print(f"  len: {len(obj)}")
        for i, item in enumerate(obj[:3]):
            itype = type(item).__name__
            ishape = getattr(item, "shape", None)
            print(f"    [{i}] type={itype}, shape={ishape}, sample={str(item)[:120]}")
        return

    # --- numpy array ---
    if isinstance(obj, np.ndarray):
        print(f"  shape : {obj.shape}")
        print(f"  dtype : {obj.dtype}")
        print(f"  min   : {obj.min():.6f}")
        print(f"  max   : {obj.max():.6f}")
        print(f"  mean  : {obj.mean():.6f}")
        if obj.ndim == 2:
            print(f"  [0,:5]: {obj[0, :5]}")
            print(f"  [-1,:5]: {obj[-1, :5]}")
        elif obj.ndim == 1:
            print(f"  [:5]  : {obj[:5]}")
        return

    # --- torch Tensor ---
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            print(f"  shape : {obj.shape}")
            print(f"  dtype : {obj.dtype}")
            print(f"  min   : {obj.min().item():.6f}")
            print(f"  max   : {obj.max().item():.6f}")
            return
    except ImportError:
        pass

    # --- fallback ---
    print(f"  repr[:200]: {repr(obj)[:200]}")


def main():
    data_dir = os.path.abspath(DATA_DIR)
    print(f"Scanning: {data_dir}\n")

    if not os.path.isdir(data_dir):
        print(f"ERROR: Directory not found: {data_dir}")
        return

    # Also list any files not in our expected list
    actual_files = [f for f in os.listdir(data_dir) if f.endswith(".pkl")]
    extra = set(actual_files) - set(FILES)
    if extra:
        print(f"Extra pickle files found: {extra}")

    for fname in FILES:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            inspect_file(fpath)
        else:
            print(f"\nMISSING: {fname}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Expected files : {len(FILES)}")
    print(f"Found          : {len([f for f in FILES if os.path.exists(os.path.join(data_dir, f))])}")
    print(f"Extra          : {len(extra)}")
    print("\nNEXT STEP: Check if any file contains drug/SE names (DataFrame index, dict keys).")
    print("If all files are plain numpy arrays, you need external mapping (see extract_mapping.py).")


if __name__ == "__main__":
    main()
