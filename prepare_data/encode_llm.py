"""
Encode drug and SE names/descriptions using PubMedBERT.
Produces:
  data/drug_llm_features.pt  — shape (757, 768)
  data/se_llm_features.pt    — shape (994, 768)
"""
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

BASE = r'd:\Đức\Đồ án thạc sĩ\MSSF\MSSF'
DATA = os.path.join(BASE, 'data')

# ─── Load mappings ────────────────────────────────────────────────────────────
drug_mapping = pd.read_csv(os.path.join(DATA, 'drug_mapping.csv'))
se_mapping = pd.read_csv(os.path.join(DATA, 'se_mapping.csv'))

print(f"Drugs: {len(drug_mapping)}, SEs: {len(se_mapping)}")
print(drug_mapping.head(3).to_string())
print(se_mapping.head(3).to_string())

# ─── Load PubMedBERT ──────────────────────────────────────────────────────────
MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
print(f"\nLoading PubMedBERT: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Device: {device}")

def encode_texts(texts, batch_size=32, max_length=128):
    """Encode list of texts to (N, 768) tensor using CLS token."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            # CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        all_embeddings.append(embeddings.cpu())
        if (i // batch_size) % 5 == 0:
            print(f"  Encoded {min(i+batch_size, len(texts))}/{len(texts)}")
    return torch.cat(all_embeddings, dim=0)

# ─── Build drug texts ─────────────────────────────────────────────────────────
# Use drug_name as input text (simple but effective for PubMedBERT)
drug_texts = drug_mapping['drug_name'].fillna('unknown drug').tolist()
print(f"\nEncoding {len(drug_texts)} drug names...")
print(f"Sample: {drug_texts[:3]}")
drug_features = encode_texts(drug_texts)
print(f"Drug features shape: {drug_features.shape}")
torch.save(drug_features, os.path.join(DATA, 'drug_llm_features.pt'))
print(f"Saved data/drug_llm_features.pt")

# ─── Build SE texts ───────────────────────────────────────────────────────────
se_texts = se_mapping['se_name'].fillna('unknown side effect').tolist()
print(f"\nEncoding {len(se_texts)} SE names...")
print(f"Sample: {se_texts[:3]}")
se_features = encode_texts(se_texts)
print(f"SE features shape: {se_features.shape}")
torch.save(se_features, os.path.join(DATA, 'se_llm_features.pt'))
print(f"Saved data/se_llm_features.pt")

print("\n=== Done! ===")
print(f"drug_llm_features: {drug_features.shape}")
print(f"se_llm_features: {se_features.shape}")
