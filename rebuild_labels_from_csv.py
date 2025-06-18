#!/usr/bin/env python3
"""
rebuild_labels_from_csv.py

Reads the balanced-test CSV (glioma_test_dataset.csv) and writes
labels.json that evaluation.py can consume.

✓ Uses EXACT basename that evaluation.py extracts with os.path.basename()
✓ Keeps the label strings exactly as they appear in the CSV
"""

import csv
import json
import os
from pathlib import Path

# ----------------------------------------------------------------------
# paths – change here if your repo layout is different
# ----------------------------------------------------------------------
LABELS_CSV = Path("data/glioma/binary/t2/glioma_dataset_all.csv")
OUT_JSON   = Path("data/glioma/binary/labels.json")

# ----------------------------------------------------------------------
# build the mapping
# ----------------------------------------------------------------------
labels = {}

with LABELS_CSV.open(newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # evaluator uses the basename of res["image_path"] as lookup key
        fname = os.path.basename(row["path"])
        labels[fname] = row["label"].strip()   # "class1", "class2", …

# ----------------------------------------------------------------------
# save
# ----------------------------------------------------------------------
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with OUT_JSON.open("w") as fp:
    json.dump(labels, fp, indent=2)

print(f"labels.json rebuilt with {len(labels):,} entries → {OUT_JSON}")