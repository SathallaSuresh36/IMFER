#!/usr/bin/env python3
"""Generate MELD metadata.csv from source annotation CSVs."""

import csv
from pathlib import Path

# Paths
data_root = Path("datasets/MELD")
output_file = data_root / "metadata.csv"

# Read all CSV files
csvs = {
    "train": "train_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv"
}

all_rows = []
for split, csv_file in csvs.items():
    csv_path = data_root / "source_annotations" / csv_file
    print(f"Reading {csv_path}...")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            all_rows.append({
                "Dialogue_ID": row.get("Dialogue_ID", ""),
                "Utterance_ID": row.get("Utterance_ID", ""),
                "Speaker": row.get("Speaker", ""),
                "Emotion": row.get("Emotion", "").lower(),
                "Utterance": row.get("Utterance", ""),
                "split": split,
            })
            count += 1
    print(f"  Added {count} utterances from {split}")

# Write combined metadata
print(f"Writing {output_file}...")
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Dialogue_ID", "Utterance_ID", "Speaker", "Emotion", "Utterance", "split"])
    writer.writeheader()
    writer.writerows(all_rows)

print(f"Created {output_file} with {len(all_rows)} total utterances")
print(f"  train: {sum(1 for r in all_rows if r['split'] == 'train')}")
print(f"  dev: {sum(1 for r in all_rows if r['split'] == 'dev')}")
print(f"  test: {sum(1 for r in all_rows if r['split'] == 'test')}")
