# MSSF-LLM: Hướng dẫn implement cho Agent Code (v2)

> **Phiên bản:** 2.0 – Viết lại hoàn toàn để khớp với cấu trúc code thực tế.
> Nguyên tắc: **Không xóa code gốc — chỉ thêm vào.**

## Mục tiêu dự án

Mở rộng mô hình MSSF gốc (Multi-Source Similarity Fusion) bằng cách thêm 3 thành phần mới:
1. **LLM branch** — PubMedBERT encode text từ DrugBank và ADReCS → vector 768-d → project xuống 384-d
2. **Cross-modal Fusion** — structured features kết hợp với LLM features qua gated fusion
3. **Supervised Contrastive Loss** — giải quyết class imbalance giữa 5 frequency classes

Repo gốc: `https://github.com/dingxlcse/MSSF.git`

### Cấu trúc code hiện tại (2 file chính)

| File | Vai trò |
|---|---|
| `model.py` | Định nghĩa kiến trúc: `Mulmodel`, `Preprocess`, `EncoderConnection`, `EncoderAddition`, `CrossProduction`, `Attention`, `GaussianParametrizer`, `Classifier` |
| `mssf.py` | Training loop: `fold_files()`, `train_test()`, `train()`, `test()`, `calculate_loss()`, `ten_fold()` |

### Data flow hiện tại

```
drug_side.pkl (757×994)
    → Extract_positive_negative_samples() → [drug_idx, se_idx, freq]
    → StratifiedKFold 10-fold
    → fold_files() → drug_train [N, 8327], side_train [N, 3976], f_train [N]
    → TensorDataset(drug, side, freq) → DataLoader
    → train():  model(batch_drug, batch_side, device) → (logits, recCon, recAdd, mu, logvar)
    → calculate_loss() → Loss = CE + 0.001*KL + 0.0001*rec1 + 0.0001*rec2
```

### Dimension flow trong `Mulmodel.forward()`

```
drugs [B, 8327]  sides [B, 3976]
    ├→ EncoderConnection  → feature1 [B, 128], recCon [B, 12303]
    ├→ EncoderAddition    → feature2 [B, 128], recAdd [B, 1751]
    └→ Preprocess + CrossProduction → feature3 [B, 128]

cat(feature1, feature2, feature3) → features [B, 384]     ← f_struct

Attention(384)      → features [B, 384]      ← đây là f_struct cuối cùng
GaussianParametrizer(384 → gp=64) → mu [B, 64], logvar [B, 64]
reparameterize      → z [B, 64]
Classifier(64 → 5)  → results [B, 5]
```

---

## Bước 0 — Clone và khảo sát repo gốc

```bash
git clone https://github.com/dingxlcse/MSSF.git
cd MSSF
```

Đọc kỹ 2 file `model.py` và `mssf.py` trước khi sửa. Xác nhận:
- `Mulmodel.forward(self, drugs, sides, device)` → return `(results, recCon, recAdd, mu, logvar)`
- `train()` unpack batch: `batch_drug, batch_side, batch_ratings = data`
- `fold_files()` return: `drug_test, side_test, f_test, drug_train, side_train, f_train`
- Labels: `batch_ratings` chứa giá trị 1-5, convert sang 0-4 bằng `(batch_ratings.long() - 1)`

---

## Bước 1 — Cài đặt thư viện bổ sung

```bash
pip install transformers torch pandas numpy scikit-learn scipy
```

Kiểm tra GPU:
```python
import torch
print(torch.cuda.is_available())
```

---

## Bước 2 — Khảo sát dữ liệu & tạo mapping (BLOCKER)

### 2.1 Inspect tất cả pickle files

Chạy script khảo sát:

```bash
python prepare_data/inspect_pickle.py
```

Script này load tất cả 13 pickle files trong `Datas/` và in:
- Type (numpy array, DataFrame, dict, ...)
- Shape, dtype
- Sample values (5 dòng đầu)

**Mục đích:** Xác định xem có file nào chứa tên thuốc / tên SE (qua DataFrame index, dict keys).

### 2.2 Trích xuất hoặc tìm mapping

```bash
python prepare_data/extract_mapping.py
```

Script này thử tự động tìm mapping từ pickle metadata. Nếu tất cả pickle files chỉ là numpy array (không có tên), script sẽ in 3 phương án thay thế:

| Phương án | Mô tả |
|---|---|
| **(A) Liên hệ tác giả** | Email dingxlcse@csu.edu.cn, xin file mapping drug_index→name và SE_index→name |
| **(B) Cross-reference** | Tải SIDER/DrugBank dataset gốc, so sánh ma trận drug_side.pkl với SIDER matrix để match thứ tự |
| **(C) Reverse-lookup** | Load `drug_mol.pkl` (Mol2Vec), tải SMILES từ DrugBank → encode Mol2Vec → cosine match |

### 2.3 Output cần có

Tạo 2 file mapping:
- `data/drug_mapping.csv` — cột: `idx`, `drugbank_id`, `drug_name` (757 dòng)
- `data/se_mapping.csv` — cột: `idx`, `adrecs_id`, `se_name` (994 dòng)

### ⚠ CHECKPOINT

> **Nếu không tạo được mapping → DỪNG LẠI. Không thể tiếp tục Bước 3.**
> Tất cả bước sau đều phụ thuộc vào việc biết tên thuốc/SE để match với DrugBank XML và ADReCS.

---

## Bước 3 — Chuẩn bị dữ liệu text (chạy 1 lần, lưu lại)

### 3.1 Parse DrugBank XML

Tải file `drugbank_all_full_database.xml` từ https://go.drugbank.com/releases (cần academic license).

Tạo file `prepare_data/parse_drugbank.py`:

```python
import xml.etree.ElementTree as ET
import pandas as pd

ns = {'db': 'http://www.drugbank.ca'}
tree = ET.parse('drugbank_all_full_database.xml')
root = tree.getroot()

records = []
for drug in root.findall('db:drug', ns):
    dbid = drug.findtext('db:drugbank-id[@primary="true"]', namespaces=ns)
    name = drug.findtext('db:name', namespaces=ns)
    desc = drug.findtext('db:description', namespaces=ns)
    moa  = drug.findtext('db:mechanism-of-action', namespaces=ns)
    records.append({'drugbank_id': dbid, 'name': name, 'description': desc, 'moa': moa})

df = pd.DataFrame(records)
df.to_csv('data/drugbank_text.csv', index=False)
print(f"Parsed {len(df)} drugs")
```

### 3.2 Parse ADReCS

Tải file `ADReCS_term.txt` từ https://bioinf.xmu.edu.cn/ADReCS.

Tạo file `prepare_data/parse_adrecs.py`:

```python
import pandas as pd

df = pd.read_csv('ADReCS_term.txt', sep='\t')
# Kiểm tra cột: print(df.columns.tolist())
# Cột thường có: ADR_ID, ADR_NAME, DEFINITION
df.to_csv('data/adrecs_text.csv', index=False)
print(f"Parsed {len(df)} side effects")
```

### 3.3 Ghép với mapping (dùng file từ Bước 2)

Tạo file `prepare_data/build_texts.py`:

```python
import pandas as pd

# Load mapping từ Bước 2
drug_map = pd.read_csv('data/drug_mapping.csv')   # idx, drugbank_id, drug_name
se_map   = pd.read_csv('data/se_mapping.csv')     # idx, adrecs_id, se_name

# Load text đã parse
drugbank = pd.read_csv('data/drugbank_text.csv')
adrecs   = pd.read_csv('data/adrecs_text.csv')

# --- Ghép thuốc ---
drugs = drug_map.merge(drugbank, left_on='drugbank_id', right_on='drugbank_id', how='left')

def make_drug_text(row):
    parts = []
    if pd.notna(row.get('description')): parts.append(str(row['description'])[:800])
    if pd.notna(row.get('moa')):         parts.append(str(row['moa'])[:400])
    return ' '.join(parts) if parts else ''

drugs['drug_text'] = drugs.apply(make_drug_text, axis=1)
drugs[['idx', 'drug_name', 'drug_text']].to_csv('data/drug_texts.csv', index=False)

matched_drugs = (drugs['drug_text'] != '').sum()
print(f"Drug texts matched: {matched_drugs}/757 ({matched_drugs/757*100:.1f}%)")

# --- Ghép side effects ---
ses = se_map.merge(adrecs, left_on='se_name', right_on='ADR_NAME', how='left')

def make_se_text(row):
    name = str(row['se_name'])
    defn = row.get('DEFINITION', '')
    if pd.notna(defn) and str(defn).strip():
        return f"{name}: {str(defn)[:600]}"
    return name  # fallback: chỉ dùng tên

ses['se_text'] = ses.apply(make_se_text, axis=1)
ses[['idx', 'se_name', 'se_text']].to_csv('data/se_texts.csv', index=False)

matched_ses = (ses['se_text'].str.len() > len(ses['se_name'].iloc[0]) + 2).sum() if len(ses) > 0 else 0
print(f"SE texts with definition: {matched_ses}/994 ({matched_ses/994*100:.1f}%)")
```

### 3.4 Encode bằng PubMedBERT (chạy 1 lần)

Tạo file `prepare_data/encode_llm.py`:

```python
import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# FREEZE toàn bộ — không train lại PubMedBERT
for param in model.parameters():
    param.requires_grad = False
model.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

def encode_texts(texts, batch_size=32):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i:i+batch_size])
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=512, return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            out = model(**inputs)
        cls = out.last_hidden_state[:, 0, :].cpu()  # [CLS] token → (batch, 768)
        all_vecs.append(cls)
        print(f"  {min(i+batch_size, len(texts))}/{len(texts)}")
    return torch.cat(all_vecs, dim=0)

os.makedirs('data', exist_ok=True)

# ---- Drug texts ----
drug_df = pd.read_csv('data/drug_texts.csv')
drug_texts = drug_df['drug_text'].fillna('').tolist()

# Track which entries have real text vs empty
drug_has_text = [1.0 if t.strip() else 0.0 for t in drug_texts]
drug_mask = torch.tensor(drug_has_text)  # (757,)

print(f"Encoding drug texts ({sum(drug_has_text):.0f}/{len(drug_texts)} have text)...")
drug_vecs = encode_texts(drug_texts)

# Replace empty-string vectors with mean of real vectors
if drug_mask.sum() > 0 and drug_mask.sum() < len(drug_mask):
    real_mean = drug_vecs[drug_mask.bool()].mean(dim=0)
    drug_vecs[~drug_mask.bool()] = real_mean
    print(f"  Replaced {(~drug_mask.bool()).sum().item()} empty vectors with mean")

torch.save(drug_vecs, 'data/drug_llm_features.pt')
torch.save(drug_mask, 'data/drug_text_mask.pt')
print(f"Drug vectors: {drug_vecs.shape}")  # Expected: (757, 768)

# ---- SE texts ----
se_df = pd.read_csv('data/se_texts.csv')
se_texts = se_df['se_text'].fillna('').tolist()

se_has_text = [1.0 if t.strip() else 0.0 for t in se_texts]
se_mask = torch.tensor(se_has_text)  # (994,)

print(f"Encoding SE texts ({sum(se_has_text):.0f}/{len(se_texts)} have text)...")
se_vecs = encode_texts(se_texts)

if se_mask.sum() > 0 and se_mask.sum() < len(se_mask):
    real_mean = se_vecs[se_mask.bool()].mean(dim=0)
    se_vecs[~se_mask.bool()] = real_mean
    print(f"  Replaced {(~se_mask.bool()).sum().item()} empty vectors with mean")

torch.save(se_vecs, 'data/se_llm_features.pt')
torch.save(se_mask, 'data/se_text_mask.pt')
print(f"SE vectors: {se_vecs.shape}")  # Expected: (994, 768)
```

Chạy theo thứ tự:
```bash
python prepare_data/parse_drugbank.py
python prepare_data/parse_adrecs.py
python prepare_data/build_texts.py
python prepare_data/encode_llm.py
```

Kiểm tra output:
- `data/drug_llm_features.pt` — shape `(757, 768)`
- `data/se_llm_features.pt` — shape `(994, 768)`
- `data/drug_text_mask.pt` — shape `(757,)`, sum > 0
- `data/se_text_mask.pt` — shape `(994,)`, sum > 0

---

## Bước 4 — Tạo các file module mới

> **Cấu trúc:** Tạo file ngang hàng với `model.py` (flat structure, không tạo package).

### 4.1 File `llm_branch.py` (đã tạo sẵn)

```python
# llm_branch.py – PubMedBERT feature projection branch.
# Input:  drug_vec (batch, 768) + se_vec (batch, 768)
# Output: (batch, 384) – khớp với MSSF fusion dim

import torch
import torch.nn as nn

class LLMBranch(nn.Module):

    def __init__(self, input_dim=768, output_dim=384, dropout=0.1):
        super().__init__()
        hidden = (input_dim + output_dim) // 2  # 576
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
        )
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, drug_vec, se_vec, drug_mask=None, se_mask=None):
        """
        drug_vec  : (batch, 768)
        se_vec    : (batch, 768)
        drug_mask : (batch,) optional – 1.0 nếu có text, 0.0 nếu không
        se_mask   : (batch,) optional
        return    : (batch, 384)
        """
        if drug_mask is not None:
            drug_vec = drug_vec * drug_mask.unsqueeze(1)
        if se_mask is not None:
            se_vec = se_vec * se_mask.unsqueeze(1)
        combined = drug_vec + se_vec
        out = self.projection(combined)
        return self.layer_norm(out)
```

**Key:** `output_dim=384` để khớp với `f_struct` sau self-attention (128×3 = 384).

### 4.2 File `cross_modal.py` (đã tạo sẵn — 2 variant)

**Variant A — Gated Fusion (mặc định):**

```python
class CrossModalGatedFusion(nn.Module):
    """
    gate   = σ( W_g · [f_struct ; f_llm] + b_g )
    f_fused = gate * f_struct + (1 − gate) * W_v(f_llm) + f_struct  (residual)

    Bias khởi tạo = -2.0 → sigmoid ≈ 0.12 ban đầu → model gần như baseline lúc đầu.
    """
    def __init__(self, d_structured=384, d_llm=384, d_model=384):
        super().__init__()
        self.gate_proj = nn.Linear(d_structured + d_llm, d_model)
        self.value_proj = nn.Linear(d_llm, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        nn.init.constant_(self.gate_proj.bias, -2.0)

    def forward(self, f_structured, f_llm):
        gate = torch.sigmoid(self.gate_proj(torch.cat([f_structured, f_llm], dim=1)))
        v_llm = self.value_proj(f_llm)
        fused = gate * f_structured + (1.0 - gate) * v_llm
        return self.layer_norm(fused + f_structured)
```

**Variant B — Multi-head Cross-Attention (ablation):**

```python
class CrossModalMultiHeadAttention(nn.Module):
    """
    Reshape f_struct (B, 384) → (B, 3, 128), coi mỗi encoder output là 1 token.
    Q = structured tokens, K/V = LLM token.
    Multi-head attention: mỗi encoder "hỏi" LLM bao nhiêu thông tin cần lấy.
    """
    # Xem file cross_modal.py để biết chi tiết implementation.
```

### 4.3 File `contrastive.py` (đã tạo sẵn — có edge case guard)

```python
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: (batch, d_model) – f_fused
        labels:   (batch,)         – 0,1,2,3,4

        Guard: Nếu class chỉ có 1 sample trong batch → skip sample đó (loss=0).
        Guard: Nếu batch_size ≤ 1 → return 0.
        """
        # ... (xem file contrastive.py cho implementation đầy đủ)
```

---

## Bước 5 — Sửa `mssf.py`: data loading & training loop

> Tất cả thay đổi ở dưới đều sửa trực tiếp trong `mssf.py`.
> Không tạo file `dataset.py` hay `train.py` riêng.

### 5.1 Thêm imports ở đầu file

```python
# ===== THÊM MỚI: Imports cho MSSF-LLM =====
from contrastive import SupervisedContrastiveLoss
```

### 5.2 Thêm args mới vào `main()`

Tìm `parser = argparse.ArgumentParser()` trong hàm `main()`. Thêm sau các args hiện có:

```python
    # ===== THÊM MỚI: MSSF-LLM arguments =====
    parser.add_argument('--use_llm', action='store_true', default=False,
                        help='Enable LLM branch (PubMedBERT features)')
    parser.add_argument('--use_cross_modal', action='store_true', default=False,
                        help='Enable cross-modal fusion')
    parser.add_argument('--use_supcon', action='store_true', default=False,
                        help='Enable supervised contrastive loss')
    parser.add_argument('--alpha_supcon', type=float, default=0.1,
                        help='Weight of supervised contrastive loss')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for SupCon loss')
    parser.add_argument('--cross_modal_variant', type=str, default='gated',
                        choices=['gated', 'multihead'],
                        help='Cross-modal fusion variant')
```

### 5.3 Sửa `fold_files()` — thêm load LLM features

Tìm hàm `fold_files()`. Sửa phần cuối, **TRƯỚC** `return`, thêm:

```python
def fold_files(data_train, data_test, args):
    # ===== CODE GỐC GIỮ NGUYÊN (đến hết f_train = data_train[:, 2]) =====

    rawdata_dir = args.rawpath
    data_train = np.array(data_train)
    data_test = np.array(data_test)
    drug_features, side_features = read_raw_data(rawdata_dir, data_test)

    drug_features_matrix = drug_features[0]
    for i in range(1, len(drug_features)):
        drug_features_matrix = np.hstack((drug_features_matrix, drug_features[i]))

    side_features_matrix = side_features[0]
    for i in range(1, len(side_features)):
        side_features_matrix = np.hstack((side_features_matrix, side_features[i]))

    drug_test = drug_features_matrix[data_test[:, 0]]
    side_test = side_features_matrix[data_test[:, 1]]
    f_test = data_test[:, 2]

    drug_train = drug_features_matrix[data_train[:, 0]]
    side_train = side_features_matrix[data_train[:, 1]]
    f_train = data_train[:, 2]

    # ===== THÊM MỚI: Load LLM features =====
    if args.use_llm:
        import torch as _torch
        drug_llm_all = _torch.load('data/drug_llm_features.pt', weights_only=True)  # (757, 768)
        se_llm_all   = _torch.load('data/se_llm_features.pt', weights_only=True)    # (994, 768)

        # Index bằng drug_idx / se_idx — giống hệt cách code gốc index features
        drug_llm_train = drug_llm_all[data_train[:, 0]].numpy()  # (N_train, 768)
        se_llm_train   = se_llm_all[data_train[:, 1]].numpy()    # (N_train, 768)
        drug_llm_test  = drug_llm_all[data_test[:, 0]].numpy()   # (N_test, 768)
        se_llm_test    = se_llm_all[data_test[:, 1]].numpy()     # (N_test, 768)

        return (drug_test, side_test, f_test,
                drug_train, side_train, f_train,
                drug_llm_train, se_llm_train, drug_llm_test, se_llm_test)

    return drug_test, side_test, f_test, drug_train, side_train, f_train
```

### 5.4 Sửa `train_test()` — TensorDataset 5 fields khi use_llm

Tìm hàm `train_test()`. Sửa phần tạo dataset:

```python
def train_test(data_train, data_test, args, fold):
    # ===== THÊM MỚI: Unpack kết quả fold_files tùy mode =====
    fold_result = fold_files(data_train, data_test, args)

    if args.use_llm:
        (drug_test, side_test, f_test,
         drug_train, side_train, f_train,
         drug_llm_train, se_llm_train, drug_llm_test, se_llm_test) = fold_result

        # TensorDataset 5 fields: drug, side, freq, drug_llm, se_llm
        trainset = torch.utils.data.TensorDataset(
            torch.FloatTensor(drug_train), torch.FloatTensor(side_train),
            torch.FloatTensor(f_train),
            torch.FloatTensor(drug_llm_train), torch.FloatTensor(se_llm_train))

        testset = torch.utils.data.TensorDataset(
            torch.FloatTensor(drug_test), torch.FloatTensor(side_test),
            torch.FloatTensor(f_test),
            torch.FloatTensor(drug_llm_test), torch.FloatTensor(se_llm_test))
    else:
        drug_test, side_test, f_test, drug_train, side_train, f_train = fold_result

        trainset = torch.utils.data.TensorDataset(
            torch.FloatTensor(drug_train), torch.FloatTensor(side_train),
            torch.FloatTensor(f_train))

        testset = torch.utils.data.TensorDataset(
            torch.FloatTensor(drug_test), torch.FloatTensor(side_test),
            torch.FloatTensor(f_test))

    # ===== Batch size: tăng lên nếu dùng SupCon =====
    train_batch = args.batch_size
    if args.use_supcon and args.batch_size < 128:
        train_batch = 128  # SupCon cần batch lớn hơn để đủ positive pairs

    _train = torch.utils.data.DataLoader(trainset, batch_size=train_batch,
                                          shuffle=True, pin_memory=False)
    _test = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                        shuffle=True, pin_memory=False)

    # ===== CODE GỐC: Device setup, model, optimizer — GIỮ NGUYÊN =====
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = Mulmodel(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ===== THÊM MỚI: SupCon loss function =====
    supcon_loss_fn = None
    if args.use_supcon:
        supcon_loss_fn = SupervisedContrastiveLoss(temperature=args.temperature)

    # ===== THÊM MỚI: Learning rate scheduler =====
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ===== CODE GỐC: Checkpoint setup — GIỮ NGUYÊN =====
    ckpt_dir = _ckpt_dir(fold)
    last_path = os.path.join(ckpt_dir, "last.pt")
    best_path = os.path.join(ckpt_dir, "best.pt")

    start_epoch = 1
    best_acc = 0.0

    if os.path.exists(last_path):
        ckpt = load_ckpt(last_path, model, optimizer, device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        print(f"[fold {fold}] RESUME from epoch {start_epoch}, best_acc={best_acc}")

    # Initialize metrics
    acc_tested = wf1_tested = maf1_tested = ka_tested = 0
    mcc_tested = maprec_tested = mareca_tested = maaupr_tested = 0

    for epoch in range(start_epoch, args.epochs + 1):
        # ===== SỬA: Truyền thêm args và supcon_loss_fn =====
        train(model, _train, optimizer, device, args, supcon_loss_fn)
        scheduler.step()

        acc_tr,weighted_f1_tr,macro_f1_tr,kappa_tr,mcc_tr,rating_tr,pred_tr,macro_prec_tr,macro_recall_tr,macro_aupr_tr = test(model, _train, device, args)
        acc_te,weighted_f1_te,macro_f1_te,kappa_te,mcc_te,rating_te,pred_te,macro_prec_te,macro_recall_te,macro_aupr_te = test(model, _test, device, args)

        if acc_te > acc_tested:
            acc_tested = acc_te
            wf1_tested = weighted_f1_te
            maf1_tested = macro_f1_te
            ka_tested = kappa_te
            mcc_tested = mcc_te
            maprec_tested = macro_prec_te
            mareca_tested = macro_recall_te
            maaupr_tested = macro_aupr_te

        save_ckpt(last_path, model, optimizer, epoch, best_acc)
        if acc_te > best_acc:
            best_acc = acc_te
            save_ckpt(best_path, model, optimizer, epoch, best_acc)

        print("Epoch: %d <Train> acc: %.5f, weighted_f1: %.5f, macro_f1: %.5f, kappa: %.5f, mcc: %.5f, precision: %.5f, recall: %.5f, aupr: %.5f" % (
            epoch, acc_tr, weighted_f1_tr, macro_f1_tr, kappa_tr, mcc_tr, macro_prec_tr, macro_recall_tr, macro_aupr_tr))
        print("Epoch: %d <Test>  acc: %.5f, weighted_f1: %.5f, macro_f1: %.5f, kappa: %.5f, mcc: %.5f, precision: %.5f, recall: %.5f, aupr: %.5f" % (
            epoch, acc_te, weighted_f1_te, macro_f1_te, kappa_te, mcc_te, macro_prec_te, macro_recall_te, macro_aupr_te))

    print(" <Best Test> acc: %.5f, weighted_f1: %.5f, macro_f1: %.5f, kappa: %.5f, mcc: %.5f, precision: %.5f, recall: %.5f, aupr: %.5f" % (
        acc_tested, wf1_tested, maf1_tested, ka_tested, mcc_tested, maprec_tested, mareca_tested, maaupr_tested))

    return acc_tested, wf1_tested, maf1_tested, ka_tested, mcc_tested, maprec_tested, mareca_tested, maaupr_tested
```

### 5.5 Sửa `train()` — unpack LLM features, tính SupCon loss

```python
def train(model, train_loader, optimizer, device, args=None, supcon_loss_fn=None):
    model.train()
    avg_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        # ===== THÊM MỚI: Unpack tùy mode =====
        if args is not None and args.use_llm:
            batch_drug, batch_side, batch_ratings, batch_drug_llm, batch_se_llm = data
        else:
            batch_drug, batch_side, batch_ratings = data
            batch_drug_llm = batch_se_llm = None

        optimizer.zero_grad()

        # ===== SỬA: Truyền thêm LLM features =====
        if batch_drug_llm is not None:
            outputs = model(batch_drug, batch_side, device, batch_drug_llm, batch_se_llm)
            multi_pred, recCon, recAdd, mu, logvar, f_fused = outputs
        else:
            multi_pred, recCon, recAdd, mu, logvar = model(batch_drug, batch_side, device)
            f_fused = None

        # ===== SỬA: Tính loss với SupCon =====
        loss = calculate_loss(multi_pred, recCon, recAdd, mu, logvar,
                              batch_ratings, batch_drug, batch_side, device,
                              f_fused=f_fused, supcon_loss_fn=supcon_loss_fn,
                              alpha_supcon=args.alpha_supcon if args else 0.0)

        loss.backward(retain_graph=True)
        optimizer.step()
        avg_loss += loss.item()

    return 0
```

### 5.6 Sửa `test()` — unpack LLM features

```python
def test(model, test_loader, device, args=None):
    model.eval()
    pred_all = []
    multi_label_all = []
    prob_all = []

    for data in test_loader:
        # ===== THÊM MỚI: Unpack tùy mode =====
        if args is not None and args.use_llm:
            test_drug, test_side, test_ratings, test_drug_llm, test_se_llm = data
        else:
            test_drug, test_side, test_ratings = data
            test_drug_llm = test_se_llm = None

        # ===== SỬA: Truyền thêm LLM features =====
        if test_drug_llm is not None:
            outputs = model(test_drug, test_side, device, test_drug_llm, test_se_llm)
            multi_pred = outputs[0]
        else:
            multi_pred, recCon, recAdd, mu, logvar = model(test_drug, test_side, device)

        pred = torch.argmax(multi_pred.cpu(), dim=1).numpy()
        pred_all.append(list(pred))
        multi_label_all.append(list((test_ratings.long()-1).cpu().numpy()))

        softmax = torch.nn.Softmax(dim=1)
        pred_prob = softmax(multi_pred).cpu().detach().numpy()
        prob_all.append(pred_prob)

    # ===== CODE GỐC: Tính metrics — GIỮ NGUYÊN =====
    pred_all = np.array(sum(pred_all, []))
    multi_label_all = np.array(sum(multi_label_all, []))
    prob_all = np.vstack(prob_all)

    acc = accuracy_score(multi_label_all, pred_all)
    weighted_f1 = f1_score(multi_label_all, pred_all, average="weighted")
    macro_f1 = f1_score(multi_label_all, pred_all, average="macro")
    kappa = cohen_kappa_score(multi_label_all, pred_all)
    mcc = matthews_corrcoef(multi_label_all, pred_all)
    macro_precision = precision_score(multi_label_all, pred_all, average='macro')
    macro_recall = recall_score(multi_label_all, pred_all, average='macro')
    multi_label_all_onehot = label_binarize(multi_label_all, classes=[0,1,2,3,4])
    macro_aupr = average_precision_score(multi_label_all_onehot, prob_all, average='macro')

    return acc, weighted_f1, macro_f1, kappa, mcc, multi_label_all, pred_all, macro_precision, macro_recall, macro_aupr
```

### 5.7 Sửa `calculate_loss()` — thêm SupCon loss

```python
def calculate_loss(multi_pred, recCon, recAdd, mu, logvar,
                   batch_ratings, batch_drug, batch_side, device,
                   f_fused=None, supcon_loss_fn=None, alpha_supcon=0.1):

    # ===== CODE GỐC — GIỮ NGUYÊN =====
    kl_div = kl_func(mu, logvar).mean()
    loss_func = nn.CrossEntropyLoss()
    multi_labels = (batch_ratings.long()-1).to(device)

    batch_vec = torch.cat((batch_drug, batch_side), dim=1)
    drug1, drug2, drug3, drug4, drug5, drug6, drug7, drug8, drug9, drug10, drug11 = batch_drug.chunk(11, 1)
    side1, side2, side3, side4 = batch_side.chunk(4, 1)
    drugs = drug1+drug2+drug3+drug4+drug5+drug6+drug7+drug8+drug9+drug10+drug11
    sides = side1+side2+side3+side4
    add_features = torch.cat((drugs, sides), dim=1)

    multi_loss = loss_func(multi_pred, multi_labels)
    reconst_loss = nn.MSELoss(reduction='none')
    rec_loss1 = reconst_loss(recCon, batch_vec.to(device)).sum(dim=-1).mean()
    rec_loss2 = reconst_loss(recAdd, add_features.to(device)).sum(dim=-1).mean()

    Loss = multi_loss + 0.001*kl_div + 0.0001*rec_loss1 + 0.0001*rec_loss2

    # ===== THÊM MỚI: Supervised Contrastive Loss =====
    if supcon_loss_fn is not None and f_fused is not None:
        L_con = supcon_loss_fn(f_fused, multi_labels)
        Loss = Loss + alpha_supcon * L_con

    return Loss
```

---

## Bước 6 — Sửa `model.py`: tích hợp LLM vào `Mulmodel`

### 6.1 Thêm imports ở đầu `model.py`

```python
# ===== THÊM MỚI =====
from llm_branch import LLMBranch
from cross_modal import CrossModalGatedFusion, CrossModalMultiHeadAttention
```

### 6.2 Sửa `Mulmodel.__init__()` — thêm LLM components

Tìm `class Mulmodel(nn.Module)`, thêm sau dòng `self.classifier = ...`:

```python
class Mulmodel(nn.Module):
    def __init__(self, args):
        super(Mulmodel, self).__init__()
        self.args = args

        # ===== CODE GỐC — GIỮ NGUYÊN =====
        self.feature_nums = 4*11
        self.encoderConnection = EncoderConnection(drugs_inputdim=757*11, sides_inputdim=994*4,
                                                    latent_dim=256, feature_dim=128, heads=4, args=args)
        self.encoderAddition = EncoderAddition(drugs_inputdim=757, sides_inputdim=994,
                                               latent_dim=256, feature_dim=128, heads=4, args=args)
        self.preprocess = Preprocess(drug_inputdim=757, side_inputdim=994, embeddim=128, args=args)
        self.crossProduction = CrossProduction(cross_dim=128, feature_dim=128, input_channel=self.feature_nums)
        self.attention = Attention(inputdim=128*3, heads=4)
        self.gaussian_parametrizer = GaussianParametrizer(feature_dim=128*3, latent_dim=args.gp)
        self.classifier = Classifier(latent_dim=args.gp, classes=5, args=args)

        # ===== THÊM MỚI: LLM branch + Cross-modal Fusion =====
        self.use_llm = getattr(args, 'use_llm', False)
        self.use_cross_modal = getattr(args, 'use_cross_modal', False)

        if self.use_llm:
            self.llm_branch = LLMBranch(input_dim=768, output_dim=128*3)  # output = 384

            if self.use_cross_modal:
                variant = getattr(args, 'cross_modal_variant', 'gated')
                if variant == 'multihead':
                    self.cross_modal = CrossModalMultiHeadAttention(
                        d_structured=384, d_llm=384, d_model=384, n_heads=4)
                else:
                    self.cross_modal = CrossModalGatedFusion(
                        d_structured=384, d_llm=384, d_model=384)
```

### 6.3 Sửa `Mulmodel.forward()` — backward-compatible

```python
    def forward(self, drugs, sides, device, drug_llm=None, se_llm=None):
        drugs = drugs.to(device)
        sides = sides.to(device)

        # ===== CODE GỐC — GIỮ NGUYÊN =====
        feature1, recCon = self.encoderConnection(drugs, sides)
        feature2, recAdd = self.encoderAddition(drugs, sides)
        drugs_pre, sides_pre = self.preprocess(drugs, sides)
        feature3 = self.crossProduction(drugs_pre, sides_pre)

        features = torch.cat((feature1, feature2, feature3), dim=1)  # [B, 384]
        features = self.attention(features)  # [B, 384]  ← f_struct

        # ===== THÊM MỚI: LLM fusion (trước BVI) =====
        f_fused = None
        if self.use_llm and drug_llm is not None and se_llm is not None:
            drug_llm = drug_llm.to(device)
            se_llm = se_llm.to(device)
            f_llm = self.llm_branch(drug_llm, se_llm)  # [B, 384]

            if self.use_cross_modal:
                features = self.cross_modal(features, f_llm)  # [B, 384]
            else:
                # Nếu chỉ dùng LLM branch (không cross-modal): cộng trực tiếp
                features = features + f_llm

            f_fused = features  # Lưu lại cho SupCon loss

        # ===== CODE GỐC — GIỮ NGUYÊN =====
        mu, logvar = self.gaussian_parametrizer(features)    # [B, 384] → [B, 64]
        latent_features = self.reparameterize(mu, logvar)    # [B, 64]
        results = self.classifier(latent_features)           # [B, 5]

        # ===== SỬA: Return thêm f_fused nếu đang dùng LLM =====
        if f_fused is not None:
            return results, recCon, recAdd, mu, logvar, f_fused

        return results, recCon, recAdd, mu, logvar
```

**Backward-compatible:** Khi `drug_llm=None` (baseline mode), forward() hoạt động y hệt code gốc, return 5 values.

---

## Bước 7 — Unit tests

### Test 1: LLM Branch

```python
import torch
from llm_branch import LLMBranch

branch = LLMBranch(input_dim=768, output_dim=384)
drug_vec = torch.randn(4, 768)
se_vec = torch.randn(4, 768)
out = branch(drug_vec, se_vec)
assert out.shape == (4, 384), f"Expected (4,384), got {out.shape}"
print("LLMBranch OK:", out.shape)

# Test with mask
mask = torch.tensor([1.0, 1.0, 0.0, 1.0])
out_masked = branch(drug_vec, se_vec, drug_mask=mask)
assert out_masked.shape == (4, 384)
print("LLMBranch with mask OK")
```

### Test 2: Cross-modal Gated Fusion

```python
from cross_modal import CrossModalGatedFusion, CrossModalMultiHeadAttention

# Variant A
gated = CrossModalGatedFusion(d_structured=384, d_llm=384, d_model=384)
f_struct = torch.randn(4, 384)
f_llm = torch.randn(4, 384)
f_fused = gated(f_struct, f_llm)
assert f_fused.shape == (4, 384), f"Expected (4,384), got {f_fused.shape}"
print("CrossModalGatedFusion OK:", f_fused.shape)

# Variant B
mha = CrossModalMultiHeadAttention(d_structured=384, d_llm=384, d_model=384, n_heads=4)
f_fused2 = mha(f_struct, f_llm)
assert f_fused2.shape == (4, 384), f"Expected (4,384), got {f_fused2.shape}"
print("CrossModalMultiHeadAttention OK:", f_fused2.shape)
```

### Test 3: SupCon Loss (bao gồm edge cases)

```python
from contrastive import SupervisedContrastiveLoss

loss_fn = SupervisedContrastiveLoss(temperature=0.1)

# Normal case
features = torch.randn(16, 384)
labels = torch.tensor([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4])
loss = loss_fn(features, labels)
assert not torch.isnan(loss), "Loss is NaN!"
print(f"SupConLoss normal: {loss.item():.4f}")

# Edge case: class with 1 sample
labels_edge = torch.tensor([0,1,1,2,2,3,3,4])
features_edge = torch.randn(8, 384)
loss_edge = loss_fn(features_edge, labels_edge)
assert not torch.isnan(loss_edge), "Loss is NaN with singleton class!"
print(f"SupConLoss edge case: {loss_edge.item():.4f}")

# Edge case: batch=1
loss_single = loss_fn(torch.randn(1, 384), torch.tensor([0]))
assert loss_single.item() == 0.0, "Batch=1 should return 0"
print("SupConLoss batch=1 OK")
```

### Test 4: Full Mulmodel forward pass

```python
import argparse
from model import Mulmodel

# Tạo args giả
args = argparse.Namespace(
    gp=64, dropout=0.4,
    use_llm=True, use_cross_modal=True, cross_modal_variant='gated'
)

model = Mulmodel(args)
device = torch.device('cpu')

batch_drug = torch.randn(8, 757*11)    # 8327
batch_side = torch.randn(8, 994*4)     # 3976
drug_llm = torch.randn(8, 768)
se_llm = torch.randn(8, 768)

# Test LLM mode
outputs = model(batch_drug, batch_side, device, drug_llm, se_llm)
results, recCon, recAdd, mu, logvar, f_fused = outputs
assert results.shape == (8, 5), f"results: {results.shape}"
assert f_fused.shape == (8, 384), f"f_fused: {f_fused.shape}"
print("Full forward (LLM mode) OK")

# Test baseline mode (backward-compatible)
args2 = argparse.Namespace(gp=64, dropout=0.4, use_llm=False, use_cross_modal=False)
model2 = Mulmodel(args2)
outputs2 = model2(batch_drug, batch_side, device)
assert len(outputs2) == 5, f"Baseline should return 5 values, got {len(outputs2)}"
print("Full forward (baseline mode) OK")
```

---

## Bước 8 — Chạy thử 1 epoch

```bash
# Baseline (code gốc, không LLM) — phải chạy bình thường
python mssf.py --epochs 1

# LLM mode (cần data/drug_llm_features.pt và data/se_llm_features.pt)
python mssf.py --epochs 1 --use_llm

# Full mode
python mssf.py --epochs 1 --use_llm --use_cross_modal --use_supcon
```

Kiểm tra:
- [ ] Baseline chạy không lỗi (backward-compatible)
- [ ] Loss không NaN
- [ ] Loss giảm (không tăng liên tục)
- [ ] L_con có giá trị hợp lý (0.1 ~ 3.0)
- [ ] GPU memory không OOM

---

## Bước 9 — Thí nghiệm chính & Evaluation

### 9.1 Metric chính

| Metric | Vai trò |
|---|---|
| **macro-AUPR** (primary) | Công bằng nhất cho class imbalance — tính AP cho mỗi class rồi lấy trung bình |
| macro-F1 (secondary) | Đánh giá balance giữa precision và recall |
| Accuracy (secondary) | Tổng quan nhưng bị bias bởi majority class |
| MCC (secondary) | Robust metric cho multi-class |

### 9.2 Ablation study — 4 variants

```bash
# Variant 1: Baseline MSSF gốc
python mssf.py --epochs 200

# Variant 2: MSSF + LLM (chỉ cộng trực tiếp, không cross-modal)
python mssf.py --epochs 200 --use_llm

# Variant 3: MSSF + LLM + Cross-modal Gated Fusion
python mssf.py --epochs 200 --use_llm --use_cross_modal

# Variant 4: Full MSSF-LLM (tất cả)
python mssf.py --epochs 200 --use_llm --use_cross_modal --use_supcon
```

### 9.3 Hyperparameter tuning (chạy trên fold 1 trước)

Chỉ tune trên fold 1 để tiết kiệm thời gian, sau đó dùng best config cho 10-fold:

```bash
# Grid search alpha_supcon
for alpha in 0.05 0.1 0.3; do
    python mssf.py --epochs 50 --use_llm --use_cross_modal --use_supcon \
        --alpha_supcon $alpha
done

# Grid search temperature
for temp in 0.05 0.1 0.2; do
    python mssf.py --epochs 50 --use_llm --use_cross_modal --use_supcon \
        --temperature $temp
done
```

### 9.4 Statistical significance testing

Sau khi chạy xong 10-fold cho mỗi variant, so sánh bằng paired t-test:

```python
from scipy.stats import ttest_rel
import numpy as np

# Ví dụ: so sánh macro_aupr giữa baseline và full MSSF-LLM
baseline_aupr = [0.xx, 0.xx, ...]   # 10 giá trị từ 10 folds
full_aupr     = [0.xx, 0.xx, ...]   # 10 giá trị từ 10 folds

t_stat, p_value = ttest_rel(full_aupr, baseline_aupr)
print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")
if p_value < 0.05:
    print("→ Improvement is STATISTICALLY SIGNIFICANT (p < 0.05)")
else:
    print("→ Improvement is NOT statistically significant")
```

### 9.5 Bảng so sánh kết quả (template)

```
| Model              | Acc (mean±std) | macro-F1 (mean±std) | macro-AUPR (mean±std) | MCC (mean±std) | p-value vs baseline |
|--------------------|----------------|---------------------|-----------------------|----------------|---------------------|
| MSSF (baseline)    |                |                     |                       |                | —                   |
| MSSF + LLM         |                |                     |                       |                |                     |
| MSSF + LLM + Cross |                |                     |                       |                |                     |
| MSSF-LLM (full)    |                |                     |                       |                |                     |
```

### 9.6 Sửa `ten_fold()` để aggregate kết quả

Thêm sau vòng lặp 10-fold trong hàm `ten_fold()`:

```python
    # ===== THÊM MỚI: Aggregate và in bảng kết quả =====
    metrics = np.array(all_results)  # (10, 8) — 10 folds × 8 metrics
    names = ['Acc', 'W-F1', 'Ma-F1', 'Kappa', 'MCC', 'Ma-Prec', 'Ma-Recall', 'Ma-AUPR']
    print("\n" + "=" * 70)
    print("10-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 70)
    for i, name in enumerate(names):
        print(f"  {name:12s}: {metrics[:, i].mean():.5f} ± {metrics[:, i].std():.5f}")
    print("=" * 70)
```

---

## Cấu trúc thư mục sau khi hoàn thành

```
MSSF/
├── model.py                          # SỬA (__init__ + forward)
├── mssf.py                           # SỬA (fold_files, train_test, train, test, calculate_loss, main)
├── llm_branch.py                     # TẠO MỚI — LLMBranch (768 → 384)
├── cross_modal.py                    # TẠO MỚI — GatedFusion + MultiHeadAttention
├── contrastive.py                    # TẠO MỚI — SupCon loss with edge guards
├── prepare_data/
│   ├── inspect_pickle.py             # TẠO MỚI — khảo sát pickle metadata
│   ├── extract_mapping.py            # TẠO MỚI — trích/tạo drug/SE mapping
│   ├── parse_drugbank.py             # TẠO MỚI — parse DrugBank XML
│   ├── parse_adrecs.py               # TẠO MỚI — parse ADReCS
│   ├── build_texts.py                # TẠO MỚI — ghép text với mapping
│   └── encode_llm.py                 # TẠO MỚI — PubMedBERT encode + missing data handling
├── data/
│   ├── drug_mapping.csv              # TẠO (Bước 2) — drug index ↔ name
│   ├── se_mapping.csv                # TẠO (Bước 2) — SE index ↔ name
│   ├── drug_texts.csv                # TẠO (Bước 3) — text cho mỗi thuốc
│   ├── se_texts.csv                  # TẠO (Bước 3) — text cho mỗi SE
│   ├── drug_llm_features.pt          # TẠO (Bước 3) — (757, 768)
│   ├── se_llm_features.pt            # TẠO (Bước 3) — (994, 768)
│   ├── drug_text_mask.pt             # TẠO (Bước 3) — (757,) binary mask
│   └── se_text_mask.pt               # TẠO (Bước 3) — (994,) binary mask
├── Datas/                            # GIỮ NGUYÊN — 13 pickle files gốc
├── checkpoints/                      # GIỮ NGUYÊN — model checkpoints
└── requirements.txt                  # SỬA — thêm transformers, scipy
```

---

## Checklist verification

- [ ] `python prepare_data/inspect_pickle.py` → 13 files readable, in shape + type
- [ ] `data/drug_mapping.csv` (757 rows) + `data/se_mapping.csv` (994 rows) tồn tại
- [ ] `data/drug_llm_features.pt` shape `(757, 768)`, `data/se_llm_features.pt` shape `(994, 768)`
- [ ] Unit tests (Bước 7) → all 4 tests pass
- [ ] `python mssf.py --epochs 1` → baseline chạy bình thường (backward-compatible)
- [ ] `python mssf.py --epochs 1 --use_llm --use_cross_modal --use_supcon` → 1 epoch không lỗi
- [ ] 10-fold hoàn chỉnh → bảng kết quả có mean±std + p-value
