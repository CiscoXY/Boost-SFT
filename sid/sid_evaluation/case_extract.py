import pandas as pd
import json
import random
import os
import argparse
import re
from collections import defaultdict
from pathlib import Path


def sanitize_filename(sid):
    parts = re.findall(r'<([^>]+)>', sid)
    if parts:
        return '_'.join(parts)
    else:
        return re.sub(r'[\\/:*?"<>|]', '_', sid)


def main(csv_path, jsonl_path, m, k, output_dir, seed=None, at_least_m=False):
    if seed is not None:
        random.seed(seed)
        print(f"Random seed set: {seed}")

    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)

    if 'sid' not in df.columns or 'item_id' not in df.columns:
        raise ValueError("CSV must contain 'sid' and 'item_id' columns")

    sid_to_items = defaultdict(list)
    for _, row in df.iterrows():
        sid_to_items[row['sid']].append(row['item_id'])

    print(f"Found {len(sid_to_items)} unique sids")

    if at_least_m:
        eligible_sids = [sid for sid, items in sid_to_items.items() if len(items) >= m]
        mode_str = "at least"
    else:
        eligible_sids = [sid for sid, items in sid_to_items.items() if len(items) == m]
        mode_str = "exactly"

    print(f"Filtered sids with {mode_str} {m} items: {len(eligible_sids)}")

    if len(eligible_sids) == 0:
        print(f"Warning: no sids found with {mode_str} {m} items, exiting")
        return

    if len(eligible_sids) < k:
        print(f"Warning: eligible sids count ({len(eligible_sids)}) is less than k ({k}), will sample all")
        k = len(eligible_sids)

    selected_sids = random.sample(eligible_sids, k)
    print(f"Randomly sampled {k} sids: {selected_sids[:3]}{'...' if k > 3 else ''}")

    print(f"Reading JSONL file: {jsonl_path}")
    item_metadata = {}
    missing_items = set()

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                item_id = data.get('item_id')
                if item_id is not None:
                    item_metadata[item_id] = data
            except json.JSONDecodeError as e:
                print(f"Warning: JSON parse error at line {line_num}: {e}")
                continue

    print(f"Loaded {len(item_metadata)} metadata records")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    success_count = 0
    for sid in selected_sids:
        item_ids = sid_to_items[sid]
        safe_filename = sanitize_filename(sid)
        output_path = os.path.join(output_dir, f"{safe_filename}.jsonl")

        records = []
        for item_id in item_ids:
            if item_id in item_metadata:
                records.append(item_metadata[item_id])
            else:
                missing_items.add(item_id)

        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        print(f"Generated file: {safe_filename}.jsonl (contains {len(records)} items)")
        success_count += 1

    if missing_items:
        print(f"Warning: {len(missing_items)} item_ids have no corresponding record in metadata")

    print(f"Done! Successfully generated {success_count} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Filter sids by item count from CSV, randomly sample and output corresponding metadata'
    )
    parser.add_argument('--csv', required=True, help='Input CSV file path')
    parser.add_argument('--jsonl', required=True, help='Input meta_data.jsonl file path')
    parser.add_argument('-m', type=int, required=True, help='Filter condition: item count')
    parser.add_argument('-k', type=int, required=True, help='Number of sids to randomly sample')
    parser.add_argument('--output', default='./output_sids', help='Output directory')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--at-least', action='store_true',
                       help='If set, filter sids with at least m items; otherwise exactly m')
    
    args = parser.parse_args()
    
    main(
        csv_path=args.csv,
        jsonl_path=args.jsonl,
        m=args.m,
        k=args.k,
        output_dir=args.output,
        seed=args.seed,
        at_least_m=args.at_least
    )