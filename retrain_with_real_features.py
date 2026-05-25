"""
retrain_with_real_features.py — Retrain IMFER with real text features.

Uses DistilRoBERTa (cached offline) for text embeddings instead of hash vectors.
Audio/visual marked as missing since we don't have the actual media files.
The model should learn text→emotion mapping which is the strongest signal.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

# Force offline mode
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

ROOT = Path(__file__).resolve().parent

# Model dimensions
D_TEXT = 768
D_AUDIO = 512
D_VISUAL = 256
D_K = 64
D_MODEL = 512
NUM_CLASSES = 6
CLASS_NAMES = ["happy", "sad", "neutral", "angry", "excited", "frustrated"]
MAX_TEXT_LEN = 50  # Most IEMOCAP utterances are short sentences

# Label mapping from manifest labels to model class indices
LABEL_MAP = {
    "joy": 0, "happy": 0,
    "sadness": 1, "sad": 1,
    "neutral": 2,
    "anger": 3, "angry": 3,
    "excited": 4,
    "frustrated": 5,
}

# DistilRoBERTa path (cached offline)
DISTILROBERTA_PATH = Path.home() / '.cache' / 'huggingface' / 'hub' / \
    'models--distilbert--distilroberta-base' / 'snapshots' / 'fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b'


def load_manifest():
    """Load IEMOCAP manifest and filter to valid labels."""
    manifest_path = ROOT / "datasets" / "manifests" / "iemocap" / "manifest_v1.jsonl"
    records = []
    with open(manifest_path, "r") as f:
        for line in f:
            rec = json.loads(line.strip())
            label = rec["label"].lower()
            if label in LABEL_MAP:
                rec["label_idx"] = LABEL_MAP[label]
                records.append(rec)
    return records


class TextEmotionDataset(Dataset):
    """Dataset that returns pre-extracted text features + label."""

    def __init__(self, records, text_features):
        self.records = records
        self.text_features = text_features

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        text_feat = self.text_features[idx]  # (L, 768)
        label = rec["label_idx"]
        return text_feat, label


def extract_all_text_features(records, tokenizer, model, batch_size=32):
    """Extract text features for all records using DistilRoBERTa."""
    print(f"  Extracting text features for {len(records)} utterances...")
    all_features = []

    for i in tqdm(range(0, len(records), batch_size), desc="  Batches"):
        batch_texts = [r["text"] for r in records[i:i + batch_size]]
        inputs = tokenizer(
            batch_texts, return_tensors="pt", max_length=MAX_TEXT_LEN,
            truncation=True, padding="max_length"
        )
        with torch.no_grad():
            outputs = model(**inputs)
        # (batch, L, 768)
        all_features.append(outputs.last_hidden_state)

    return torch.cat(all_features, dim=0)  # (N, L, 768)


def train():
    print("=" * 70)
    print("  IMFER Retraining with Real Text Features (DistilRoBERTa)")
    print("=" * 70)

    # Load tokenizer and model
    print("\n[1/5] Loading DistilRoBERTa from cache...")
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(str(DISTILROBERTA_PATH))
    text_model = AutoModel.from_pretrained(str(DISTILROBERTA_PATH))
    text_model.eval()
    print("  [OK] DistilRoBERTa loaded")

    # Load manifest
    print("\n[2/5] Loading IEMOCAP manifest...")
    all_records = load_manifest()
    train_records = [r for r in all_records if r["split"] == "train"]
    val_records = [r for r in all_records if r["split"] == "val"]
    test_records = [r for r in all_records if r["split"] == "test"]
    print(f"  Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")

    # Check label distribution
    from collections import Counter
    train_dist = Counter(r["label_idx"] for r in train_records)
    print(f"  Train distribution: {dict(sorted(train_dist.items()))}")

    # Extract features
    print("\n[3/5] Extracting text features...")
    train_features = extract_all_text_features(train_records, tokenizer, text_model)
    val_features = extract_all_text_features(val_records, tokenizer, text_model)
    test_features = extract_all_text_features(test_records, tokenizer, text_model)
    print(f"  [OK] Features extracted: train={train_features.shape}, val={val_features.shape}")

    # Create datasets
    train_dataset = TextEmotionDataset(train_records, train_features)
    val_dataset = TextEmotionDataset(val_records, val_features)
    test_dataset = TextEmotionDataset(test_records, test_features)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load IMFER model
    print("\n[4/5] Initializing IMFER model...")
    from models import IMFER
    model = IMFER(
        d_text=D_TEXT, d_audio=D_AUDIO, d_visual=D_VISUAL,
        d_k=D_K, d_model=D_MODEL,
        num_classes=NUM_CLASSES,
        casgt_heads=8, casgt_layers=4, context_window=10,
        dropout=0.3
    )
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # Class weights to handle imbalance
    class_counts = torch.tensor([train_dist.get(i, 1) for i in range(NUM_CLASSES)], dtype=torch.float)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"  Class weights: {class_weights.tolist()}")

    # Zero tensors for missing audio/visual
    audio_zero = torch.zeros(1, 100, D_AUDIO)
    visual_zero = torch.zeros(1, 30, D_VISUAL)

    # Training loop
    print("\n[5/5] Training...")
    best_val_acc = 0.0
    save_path = ROOT / "artifacts" / "iemocap" / "seed_42" / "checkpoints" / "best.pt"

    num_epochs = 30
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for text_feat, labels in train_loader:
            B = text_feat.shape[0]
            # text_feat: (B, L, 768)
            # Audio/visual: zeros (missing modality)
            audio_feat = audio_zero.expand(B, -1, -1)
            visual_feat = visual_zero.expand(B, -1, -1)

            optimizer.zero_grad()

            # Forward pass - single utterance (no conversation context)
            # HCMA expects (B, L, d_mod), use utterance-level directly
            logits, mcs = model.hcma_mcs_forward(text_feat, audio_feat, visual_feat)

            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * B
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += B

        scheduler.step()
        train_acc = correct / total
        train_loss = total_loss / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for text_feat, labels in val_loader:
                B = text_feat.shape[0]
                audio_feat = audio_zero.expand(B, -1, -1)
                visual_feat = visual_zero.expand(B, -1, -1)
                logits, _ = model.hcma_mcs_forward(text_feat, audio_feat, visual_feat)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += B

        val_acc = val_correct / val_total

        print(f"  Epoch {epoch:2d}/{num_epochs}: loss={train_loss:.4f} "
              f"train_acc={train_acc*100:.1f}% val_acc={val_acc*100:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"    * Best model saved (val_acc={val_acc*100:.1f}%)")

    # Test evaluation
    print(f"\n  Loading best model (val_acc={best_val_acc*100:.1f}%)...")
    model.load_state_dict(torch.load(save_path, map_location="cpu"))
    model.eval()

    test_correct = 0
    test_total = 0
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    with torch.no_grad():
        for text_feat, labels in test_loader:
            B = text_feat.shape[0]
            audio_feat = audio_zero.expand(B, -1, -1)
            visual_feat = visual_zero.expand(B, -1, -1)
            logits, _ = model.hcma_mcs_forward(text_feat, audio_feat, visual_feat)
            preds = logits.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += B
            for i in range(B):
                l = labels[i].item()
                class_total[l] += 1
                if preds[i].item() == l:
                    class_correct[l] += 1

    test_acc = test_correct / test_total
    print(f"\n{'='*70}")
    print(f"  TEST RESULTS: Overall Accuracy = {test_acc*100:.1f}%")
    print(f"{'='*70}")
    for i, cls in enumerate(CLASS_NAMES):
        if class_total[i] > 0:
            print(f"  {cls:12s}: {class_correct[i]:4d}/{class_total[i]:4d} = "
                  f"{class_correct[i]/class_total[i]*100:.1f}%")
    print(f"{'='*70}")
    print(f"\n  Checkpoint saved: {save_path}")
    print("  Done! Now run: python demo_inference.py")


if __name__ == "__main__":
    train()
