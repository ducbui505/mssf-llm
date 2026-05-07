# MSSF-LLM: Hướng dẫn implement cho Agent Code

## Mục tiêu dự án

Mở rộng mô hình MSSF gốc (Multi-Source Similarity Fusion) bằng cách thêm 3 thành phần mới:
1. **LLM branch** — PubMedBERT encode text từ DrugBank và ADReCS
2. **Cross-modal Attention** — structured features hỏi LLM features
3. **Supervised Contrastive Loss** — giải quyết class imbalance

Repo gốc: `https://github.com/dingxlcse/MSSF.git`

---

## Bước 0 — Clone và khảo sát repo gốc

```bash
git clone https://github.com/dingxlcse/MSSF.git
cd MSSF
```

Sau khi clone, liệt kê toàn bộ file Python:
```bash
find . -name "*.py" | sort
```

Đọc từng file để hiểu:
- File nào định nghĩa **kiến trúc mô hình** (class Model, class Encoder...)
- File nào là **training loop** (vòng lặp train, tính loss, optimizer.step)
- File nào là **dataset/dataloader** (load dữ liệu similarity matrices)
- File nào là **entry point** (main.py hoặc run.py)

> **Quan trọng:** Đọc kỹ trước khi sửa. Không xóa bất kỳ code gốc nào — chỉ thêm vào.

---

## Bước 1 — Cài đặt thư viện bổ sung

```bash
pip install transformers torch pandas numpy scikit-learn
```

Kiểm tra GPU:
```python
import torch
print(torch.cuda.is_available())
```

---

## Bước 2 — Chuẩn bị dữ liệu text (chạy 1 lần, lưu lại)

### 2.1 Parse DrugBank XML

Tải file `drugbank_all_full_database.xml` từ drugbank.com (cần đăng ký tài khoản).

Tạo file `prepare_data/parse_drugbank.py`:

```python
import xml.etree.ElementTree as ET
import pandas as pd

ns = {'db': 'http://www.drugbank.ca'}
tree = ET.parse('drugbank_all_full_database.xml')
root = tree.getroot()

records = []
for drug in root.findall('db:drug', ns):
    name = drug.findtext('db:name', namespaces=ns)
    desc = drug.findtext('db:description', namespaces=ns)
    moa  = drug.findtext('db:mechanism-of-action', namespaces=ns)
    records.append({'name': name, 'description': desc, 'moa': moa})

df = pd.DataFrame(records)
df.to_csv('data/drugbank_text.csv', index=False)
print(f"Parsed {len(df)} drugs")
```

### 2.2 Parse ADReCS

Tải file `ADReCS_term.txt` từ bioinf.xmu.edu.cn/ADReCS.

Tạo file `prepare_data/parse_adrecs.py`:

```python
import pandas as pd

df = pd.read_csv('ADReCS_term.txt', sep='\t')
# Giữ cột tên SE và định nghĩa lâm sàng
# Tên cột có thể là: ADR_NAME, DEFINITION, hoặc tương tự
# Kiểm tra: print(df.columns.tolist())
df.to_csv('data/adrecs_text.csv', index=False)
print(f"Parsed {len(df)} side effects")
```

### 2.3 Ghép với benchmark dataset

Trong repo MSSF gốc có danh sách 757 thuốc và 994 side effects. Tìm file danh sách này (thường là `.txt` hoặc `.csv` trong thư mục `data/`).

Tạo file `prepare_data/build_texts.py`:

```python
import pandas as pd

# Load danh sách benchmark từ repo MSSF
# Tìm file chứa tên thuốc — thường là drug_list.txt hoặc tương tự
drug_list = pd.read_csv('data/drug_list.csv')        # tên cột: drug_name
se_list   = pd.read_csv('data/se_list.csv')          # tên cột: se_name

# Load text đã parse
drugbank  = pd.read_csv('data/drugbank_text.csv')
adrecs    = pd.read_csv('data/adrecs_text.csv')

# Ghép thuốc
drugs = drug_list.merge(drugbank, left_on='drug_name', right_on='name', how='left')

def make_drug_text(row):
    parts = []
    if pd.notna(row.get('description')): parts.append(str(row['description'])[:800])
    if pd.notna(row.get('moa')):         parts.append(str(row['moa'])[:400])
    return ' '.join(parts) if parts else str(row['drug_name'])

drugs['drug_text'] = drugs.apply(make_drug_text, axis=1)
drugs[['drug_name', 'drug_text']].to_csv('data/drug_texts.csv', index=False)
print(f"Drug texts: {drugs['drug_text'].notna().sum()}/757")

# Ghép side effects
# Tên cột trong ADReCS: kiểm tra lại sau khi parse
ses = se_list.merge(adrecs, left_on='se_name', right_on='ADR_NAME', how='left')

def make_se_text(row):
    name = str(row['se_name'])
    defn = row.get('DEFINITION', '')
    if pd.notna(defn) and str(defn).strip():
        return f"{name}: {str(defn)[:600]}"
    return name

ses['se_text'] = ses.apply(make_se_text, axis=1)
ses[['se_name', 'se_text']].to_csv('data/se_texts.csv', index=False)
print(f"SE texts: {ses['se_text'].notna().sum()}/994")
```

### 2.4 Encode bằng PubMedBERT (chạy 1 lần)

Tạo file `prepare_data/encode_llm.py`:

```python
import torch
import pandas as pd
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
        cls = out.last_hidden_state[:, 0, :].cpu()  # [CLS] token
        all_vecs.append(cls)
        print(f"  {min(i+batch_size, len(texts))}/{len(texts)}")
    return torch.cat(all_vecs, dim=0)

# Encode drug texts
drug_texts = pd.read_csv('data/drug_texts.csv')['drug_text'].fillna('').tolist()
drug_vecs  = encode_texts(drug_texts)
torch.save(drug_vecs, 'data/drug_llm_features.pt')
print(f"Drug vectors: {drug_vecs.shape}")  # (757, 768)

# Encode SE texts
se_texts = pd.read_csv('data/se_texts.csv')['se_text'].fillna('').tolist()
se_vecs  = encode_texts(se_texts)
torch.save(se_vecs, 'data/se_llm_features.pt')
print(f"SE vectors: {se_vecs.shape}")  # (994, 768)
```

Chạy theo thứ tự:
```bash
python prepare_data/parse_drugbank.py
python prepare_data/parse_adrecs.py
python prepare_data/build_texts.py
python prepare_data/encode_llm.py
```

Kiểm tra output:
- `data/drug_llm_features.pt` — shape phải là `(757, 768)`
- `data/se_llm_features.pt`   — shape phải là `(994, 768)`

---

## Bước 3 — Tạo các file module mới

### 3.1 Tạo file `model/llm_branch.py`

```python
import torch
import torch.nn as nn

class LLMBranch(nn.Module):
    """
    Nhận vector PubMedBERT đã encode sẵn (768d)
    của drug và SE, ghép bằng [SEP] logic (cộng),
    chiếu xuống d_model chiều để khớp với MSSF features.
    """
    def __init__(self, input_dim=768, output_dim=64, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim)
        )
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, drug_vec, se_vec):
        """
        drug_vec: (batch, 768) — vector PubMedBERT của thuốc
        se_vec:   (batch, 768) — vector PubMedBERT của SE
        return:   (batch, output_dim)
        """
        # Kết hợp drug + SE bằng cách cộng (element-wise)
        combined = drug_vec + se_vec          # (batch, 768)
        out = self.projection(combined)       # (batch, output_dim)
        return self.layer_norm(out)
```

### 3.2 Tạo file `model/cross_modal.py`

```python
import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):
    """
    Structured features (Q) hỏi LLM features (K, V).
    F_struct: output từ Self-attention + Feed Forward của MSSF
    F_llm:    output từ LLMBranch
    """
    def __init__(self, d_structured, d_llm, d_model=64):
        super().__init__()
        self.W_Q = nn.Linear(d_structured, d_model)
        self.W_K = nn.Linear(d_llm, d_model)
        self.W_V = nn.Linear(d_llm, d_model)
        self.scale = d_model ** 0.5
        self.out_proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, f_structured, f_llm):
        """
        f_structured: (batch, d_structured)
        f_llm:        (batch, d_llm)
        return:       (batch, d_model) — F_fused
        """
        Q = self.W_Q(f_structured)
        K = self.W_K(f_llm)
        V = self.W_V(f_llm)

        # Attention score: dot product Q và K
        score = (Q * K).sum(dim=-1, keepdim=True) / self.scale  # (batch, 1)
        weight = torch.sigmoid(score)

        # Lấy V theo trọng số attention
        attended = weight * V                     # (batch, d_model)
        out = self.out_proj(attended)

        # Residual: cộng Q để không mất thông tin structured
        return self.layer_norm(out + Q)
```

### 3.3 Tạo file `model/contrastive.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    """
    Kéo các mẫu cùng frequency class lại gần nhau,
    đẩy các mẫu khác class ra xa. Giải quyết class imbalance.

    Dùng F_fused (trước BVI) làm input để tín hiệu sạch hơn.
    Loss này KHÔNG tạo ra vector mới — chỉ đóng góp vào gradient.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: (batch, d_model) — F_fused từ Cross-modal Attention
        labels:   (batch,)         — nhãn 0,1,2,3,4 (5 frequency classes)
        return:   scalar loss
        """
        # Normalize để tính cosine similarity
        features = F.normalize(features, dim=1)

        # Ma trận similarity (batch x batch)
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # Mask: cặp cùng class = 1, khác class hoặc chính nó = 0
        labels = labels.unsqueeze(1)                              # (batch, 1)
        mask_pos  = (labels == labels.T).float()                  # cùng class
        mask_self = torch.eye(features.size(0), device=features.device)
        mask_pos  = mask_pos - mask_self                          # bỏ chính nó

        # Tính log probability
        exp_sim  = torch.exp(sim_matrix - sim_matrix.max(dim=1, keepdim=True).values)
        log_prob = torch.log(exp_sim / (exp_sim.sum(dim=1, keepdim=True) + 1e-8))

        # Chỉ tính loss trên positive pairs
        n_pos = mask_pos.sum(dim=1).clamp(min=1)
        loss  = -(log_prob * mask_pos).sum(dim=1) / n_pos
        return loss.mean()
```

---

## Bước 4 — Sửa file Dataset

Tìm file dataset trong repo MSSF (thường là `dataset.py` hoặc nằm trong `utils.py`).

Thêm việc load LLM features vào DataLoader. Tìm class Dataset và thêm:

```python
# Thêm vào __init__ của Dataset class
import torch

# Load LLM features (đã encode sẵn từ Bước 2)
self.drug_llm = torch.load('data/drug_llm_features.pt')  # (757, 768)
self.se_llm   = torch.load('data/se_llm_features.pt')    # (994, 768)

# Thêm vào __getitem__ của Dataset class
# Giả sử drug_idx và se_idx là index của thuốc và SE trong batch
return {
    # ... các field hiện có của MSSF ...
    'drug_llm': self.drug_llm[drug_idx],   # (768,)
    'se_llm':   self.se_llm[se_idx],       # (768,)
}
```

> Lưu ý: Tìm đúng tên biến drug_idx và se_idx trong code gốc. Có thể tên khác nhau tùy repo.

---

## Bước 5 — Sửa file Model chính

Tìm file định nghĩa class Model chính trong repo MSSF. Thêm các import và thành phần mới:

### 5.1 Thêm import ở đầu file

```python
from model.llm_branch import LLMBranch
from model.cross_modal import CrossModalAttention
from model.contrastive import SupervisedContrastiveLoss
```

### 5.2 Thêm vào __init__ của Model class

```python
# Tìm __init__ của Model class, thêm sau phần khởi tạo BVI:

# === THÊM MỚI: LLM branch ===
self.llm_branch = LLMBranch(
    input_dim=768,
    output_dim=64    # khớp với d_model của MSSF
)

# === THÊM MỚI: Cross-modal Attention ===
# d_structured: chiều output của Feed Forward trong MSSF
# Tìm giá trị này trong code gốc (thường là 64 hoặc 128)
self.cross_attn = CrossModalAttention(
    d_structured=64,   # thay bằng chiều thực tế của f_struct
    d_llm=64,
    d_model=64
)
```

### 5.3 Sửa forward() của Model class

Tìm hàm `forward()` trong Model class. Thêm các bước sau:

```python
def forward(self, batch):
    # ===== CODE GỐC MSSF — GIỮ NGUYÊN =====
    # (copy toàn bộ forward gốc vào đây, không xóa gì)
    # Giả sử forward gốc kết thúc với:
    #   f_struct = ... (output sau Feed Forward + Add&Norm)
    #   mu, log_var, z = bvi(f_struct)
    #   logits = classifier(z)
    # ==========================================

    # === THÊM MỚI: LLM branch ===
    f_llm = self.llm_branch(
        batch['drug_llm'],   # (batch, 768)
        batch['se_llm']      # (batch, 768)
    )                        # → (batch, 64)

    # === THÊM MỚI: Cross-modal Attention ===
    # f_struct là output sau Feed Forward của MSSF gốc
    # Thêm dòng này TRƯỚC khi gọi BVI
    f_fused = self.cross_attn(f_struct, f_llm)  # → (batch, 64)

    # === SỬA: Đưa f_fused vào BVI thay vì f_struct ===
    mu, log_var, z = self.bvi(f_fused)   # thay f_struct → f_fused
    logits = self.classifier(z)

    # Return thêm f_fused để tính SupCon loss bên ngoài
    return logits, mu, log_var, f_fused
```

---

## Bước 6 — Sửa Training Loop

Tìm file `train.py` hoặc nơi tính loss trong repo gốc. Thêm SupCon loss:

```python
from model.contrastive import SupervisedContrastiveLoss

# Khởi tạo (thêm vào phần setup, trước vòng lặp train)
supcon_loss_fn = SupervisedContrastiveLoss(temperature=0.07)
alpha = 0.3   # trọng số của SupCon loss — có thể tune sau

# === SỬA training step ===
# Tìm chỗ tính loss và thêm:

def train_step(model, batch, optimizer):
    optimizer.zero_grad()

    # Forward — nhận thêm f_fused
    logits, mu, log_var, f_fused = model(batch)
    labels = batch['label']

    # Loss gốc MSSF (giữ nguyên — tìm trong code gốc)
    L_cls = classification_loss(logits, labels)    # cross entropy
    L_kl  = kl_loss(mu, log_var)                   # KL divergence

    # === THÊM MỚI: Supervised Contrastive Loss ===
    L_con = supcon_loss_fn(f_fused, labels)

    # Tổng loss
    L_total = L_cls + 0.1 * L_kl + alpha * L_con

    L_total.backward()
    optimizer.step()

    return {
        'L_total': L_total.item(),
        'L_cls':   L_cls.item(),
        'L_kl':    L_kl.item(),
        'L_con':   L_con.item()
    }
```

---

## Bước 7 — Kiểm tra từng bước

### Test 1: Module LLM branch

```python
import torch
from model.llm_branch import LLMBranch

branch = LLMBranch(input_dim=768, output_dim=64)
drug_vec = torch.randn(4, 768)   # batch=4
se_vec   = torch.randn(4, 768)
out = branch(drug_vec, se_vec)
assert out.shape == (4, 64), f"Expected (4,64), got {out.shape}"
print("LLMBranch OK:", out.shape)
```

### Test 2: Cross-modal Attention

```python
from model.cross_modal import CrossModalAttention

attn = CrossModalAttention(d_structured=64, d_llm=64, d_model=64)
f_struct = torch.randn(4, 64)
f_llm    = torch.randn(4, 64)
f_fused  = attn(f_struct, f_llm)
assert f_fused.shape == (4, 64), f"Expected (4,64), got {f_fused.shape}"
print("CrossModalAttention OK:", f_fused.shape)
```

### Test 3: SupCon Loss

```python
from model.contrastive import SupervisedContrastiveLoss

loss_fn = SupervisedContrastiveLoss(temperature=0.07)
features = torch.randn(8, 64)
labels   = torch.tensor([0, 0, 1, 1, 2, 2, 3, 4])
loss = loss_fn(features, labels)
assert not torch.isnan(loss), "Loss is NaN!"
print("SupConLoss OK:", loss.item())
```

### Test 4: Forward pass toàn bộ model

```python
# Tạo batch giả với đúng các key
batch = {
    # ... các key hiện có của MSSF batch ...
    'drug_llm': torch.randn(8, 768),
    'se_llm':   torch.randn(8, 768),
    'label':    torch.randint(0, 5, (8,))
}

model = MSSFwithLLM(config)   # tên class model trong repo gốc
logits, mu, log_var, f_fused = model(batch)
assert logits.shape  == (8, 5),  f"logits: {logits.shape}"
assert f_fused.shape == (8, 64), f"f_fused: {f_fused.shape}"
print("Full forward pass OK")
```

---

## Bước 8 — Chạy thử 1 epoch

```python
# Chạy 1 epoch với tất cả training data để check không có lỗi
python train.py --epochs 1 --debug
```

Kiểm tra:
- [ ] Loss giảm (không tăng liên tục)
- [ ] Không có NaN trong loss
- [ ] L_con có giá trị hợp lý (0.1 ~ 2.0)
- [ ] GPU memory không out-of-memory

---

## Bước 9 — Chạy thực nghiệm chính

### 9.1 Baseline: MSSF gốc (không thay đổi)

```bash
python train.py --model mssf_original
```

### 9.2 MSSF + LLM (không cross-modal)

```bash
python train.py --model mssf_llm_only --use_llm True --use_cross False --use_supcon False
```

### 9.3 MSSF + LLM + Cross-modal (không SupCon)

```bash
python train.py --model mssf_llm_cross --use_llm True --use_cross True --use_supcon False
```

### 9.4 MSSF-LLM đầy đủ (tất cả)

```bash
python train.py --model mssf_llm_full --use_llm True --use_cross True --use_supcon True
```

---

## Cấu trúc thư mục sau khi hoàn thành

```
MSSF/
├── data/
│   ├── drug_llm_features.pt      # (757, 768) — TẠO MỚI
│   ├── se_llm_features.pt        # (994, 768) — TẠO MỚI
│   ├── drug_texts.csv            # TẠO MỚI
│   ├── se_texts.csv              # TẠO MỚI
│   └── [các file data gốc MSSF]
├── model/
│   ├── llm_branch.py             # TẠO MỚI
│   ├── cross_modal.py            # TẠO MỚI
│   ├── contrastive.py            # TẠO MỚI
│   └── [các file model gốc MSSF — GIỮ NGUYÊN]
├── prepare_data/
│   ├── parse_drugbank.py         # TẠO MỚI
│   ├── parse_adrecs.py           # TẠO MỚI
│   ├── build_texts.py            # TẠO MỚI
│   └── encode_llm.py             # TẠO MỚI
├── train.py                      # SỬA (thêm SupCon loss)
├── dataset.py                    # SỬA (thêm load LLM features)
└── [model chính].py              # SỬA (thêm LLMBranch + CrossModalAttn)
```

---

## Lưu ý quan trọng cho Agent

1. **Đọc code gốc trước** — Tìm đúng tên class, tên biến, chiều d_model thực tế trong repo MSSF trước khi sửa. Các giá trị như `d_structured=64` có thể khác.

2. **Không xóa code gốc** — Chỉ thêm vào. Nếu cần sửa `forward()`, comment code cũ lại thay vì xóa.

3. **Chiều d_model** — Tìm trong code gốc chiều output thực tế của Feed Forward layer. Thay `d_structured=64` bằng giá trị đúng.

4. **Tên biến trong batch** — Tìm đúng tên `drug_idx` và `se_idx` trong Dataset gốc để load đúng LLM vector.

5. **Random seed** — Thêm vào đầu train.py:
```python
import torch, numpy as np, random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

6. **Lưu checkpoint** sau mỗi fold:
```python
torch.save(model.state_dict(), f'checkpoints/fold_{fold}.pt')
```

7. **Nếu out-of-memory** — Giảm batch size hoặc dùng `torch.cuda.empty_cache()` giữa các epoch.
