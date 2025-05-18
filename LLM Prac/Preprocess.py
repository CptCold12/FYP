import os
import json
import csv
import ast
import torch
from tqdm import tqdm
import random

filter = True
timeline_dir = r"D:\CourseworkFolder\DPSynthData\Data Manipulation\timelines"
token_map_path = r"D:\CourseworkFolder\DPSynthData\Data Manipulation\token_map.json"
train_out_path = "train.pt"
val_out_path = "val.pt"
EOS_TOKEN = "__EOS__"

REMOVED_FIELDS = {
    "intime", "outtime", "admittime", "dischtime",
    "starttime", "charttime", "chartdate", "ordertime", "valuenum", "long_title"
}

# Load the pre-built token map
with open(token_map_path, "r", encoding="utf-8") as f:
    token_map = json.load(f)

# Add EOS token if not present
if EOS_TOKEN not in token_map:
    token_map[EOS_TOKEN] = max(token_map.values()) + 1
    with open(token_map_path, "w", encoding="utf-8") as f:
        json.dump(token_map, f, indent=4)
    print(f"✅ Added '{EOS_TOKEN}' at ID {token_map[EOS_TOKEN]}")

EOS_ID = token_map[EOS_TOKEN]

all_timelines = []
timeline_files = [f for f in os.listdir(timeline_dir) if f.endswith('.csv')]
timeline_files = [os.path.join(timeline_dir, f) for f in timeline_files]

for path in tqdm(timeline_files, desc="Tokenizing Timelines"):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        timeline_sequence = []
        diagnosis_sequence = []  # To store Diagnosis events separately

        for row in reader:
            event_type = row.get("event_type")
            details_str = row.get("details")

            if not event_type or not details_str:
                continue

            try:
                details_dict = ast.literal_eval(details_str)
            except (ValueError, SyntaxError):
                continue  # Skip malformed details

            # Remove unnecessary fields
            for key in REMOVED_FIELDS:
                details_dict.pop(key, None)

            # Handle ICD codes (if present)
            if 'icd_code' in details_dict:
                code = details_dict['icd_code']
                if isinstance(code, str) and len(code) >= 3:
                    details_dict['icd_code'] = code[:3]

            # Combine event type and details to create the token string
            details_token = "_".join([f"{k}={v}" for k, v in details_dict.items()])
            combined_token = f"{event_type}_{details_token}"

            # Check if the combined token is in the token map
            if combined_token in token_map:
                token_id = token_map[combined_token]

                # Check if event_type is a Diagnosis, if so, add it to the beginning sequence
                if "Diagnosis" in event_type:
                    diagnosis_sequence.append(token_id)
                else:
                    timeline_sequence.append(token_id)

        # Prepend Diagnosis events to the main timeline sequence
        if diagnosis_sequence:
            timeline_sequence = diagnosis_sequence + timeline_sequence

        # Add EOS token at the end of valid timelines
        if timeline_sequence:
            timeline_sequence.append(EOS_ID)
            if len(timeline_sequence) < 30000 and filter:
                all_timelines.append(timeline_sequence)

            elif not filter:
                all_timelines.append(timeline_sequence)

# Shuffle and split data into training and validation sets
random.shuffle(all_timelines)
n = int(0.9 * len(all_timelines))
train_seqs = all_timelines[:n]
val_seqs = all_timelines[n:]

# Save to .pt files
torch.save(train_seqs, train_out_path)
torch.save(val_seqs, val_out_path)

print(f"Saved {train_out_path} ({len(train_seqs)} timelines), {val_out_path} ({len(val_seqs)} timelines)")
