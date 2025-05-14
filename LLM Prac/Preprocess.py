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

with open(token_map_path, "r", encoding="utf-8") as f:
    token_map = json.load(f)

if EOS_TOKEN not in token_map:
    token_map[EOS_TOKEN] = max(token_map.values()) + 1
    with open(token_map_path, "w", encoding="utf-8") as f:
        json.dump(token_map, f, indent=4)
    print(f"✅ Added '{EOS_TOKEN}' at ID {token_map[EOS_TOKEN]}")

EOS_ID = token_map[EOS_TOKEN]

all_timelines = []
count = 0
timeline_files = [f for f in os.listdir(timeline_dir) if f.endswith('.csv')]
timeline_files = [os.path.join(timeline_dir, f) for f in timeline_files]

for path in tqdm(timeline_files, desc="Tokenizing Timelines"):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        timeline_sequence = []

        for row in reader:
            event_type = row.get("event_type")
            details_str = row.get("details")
    

            if not event_type or not details_str:
                continue

            try:
                details_dict = ast.literal_eval(details_str)
                
            except:
                continue

            for key in REMOVED_FIELDS:
                details_dict.pop(key, None)
                
            if 'icd_code' in details_dict:
                    code = details_dict['icd_code']
                    if isinstance(code, str) and len(code) >= 3:
                        details_dict['icd_code'] = code[:3]

            event_type_token = event_type
            details_token = "_".join([f"{k}={v}" for k, v in details_dict.items()])
            if event_type_token in token_map and details_token in token_map:
                timeline_sequence.append(token_map[event_type_token])
                timeline_sequence.append(token_map[details_token])

        if timeline_sequence:
            timeline_sequence.append(EOS_ID)
            if len(timeline_sequence) < 30000 and filter:
                all_timelines.append(timeline_sequence)
                print()
            
            elif not filter:
                all_timelines.append(timeline_sequence)
            
        print(all_timelines)

random.shuffle(all_timelines)
n = int(0.9 * len(all_timelines))
train_seqs = all_timelines[:n]
val_seqs = all_timelines[n:]

torch.save(train_seqs, train_out_path)
torch.save(val_seqs, val_out_path)

print(f"Saved {train_out_path} ({len(train_seqs)} timelines), {val_out_path} ({len(val_seqs)} timelines)")
