import pandas as pd
import numpy as np
import re
import os
import json
from collections import defaultdict

GROUNDTRUTH_PATH = '/path/to/groundtruth/dataset.csv'
SID_ITEM_MAP_PATH = '/path/to/sid_map/itemid_to_sid.csv'

JSONL_DIR = '/path/to/generated/experiment_output'
OUTPUT_METRICS_PATH = '/path/to/generated/experiment_output/metrics_summary.csv'
OUTPUT_STATS_PATH = '/path/to/generated/experiment_output/stats_summary.csv'

K_VALUES = [1, 3, 5, 10, 20]

SID_PATTERN = re.compile(r'<a_(\d{1,3})><b_(\d{1,3})><c_(\d{1,3})>')
MAX_ITEM_PER_SID = 3
MAX_PREDICT_LEN = 20

def calculate_hit_rate(predicted_items, true_item, k):
    if k <= 0:
        return np.nan
    pred_k = predicted_items[:k]
    return 1.0 if true_item in pred_k else 0.0

def calculate_ndcg(predicted_items, true_item, k):
    if k <= 0:
        return np.nan

    pred_k = predicted_items[:k]
    relevance = np.array([1 if item == true_item else 0 for item in pred_k])

    if true_item not in pred_k:
        return 0.0

    i_plus_1 = np.arange(1, len(relevance) + 1)
    denominator = np.log2(i_plus_1 + 1)
    dcg = np.sum(relevance / denominator)
    idcg = 1.0

    return dcg / idcg

def load_data():
    gt_df = pd.read_csv(GROUNDTRUTH_PATH)
    if 'item_id' not in gt_df.columns:
        raise ValueError(f"Groundtruth CSV must contain 'item_id' column.")

    map_df = pd.read_csv(SID_ITEM_MAP_PATH, dtype={'item_id': 'int', 'sid': 'str'})
    sid_to_items = map_df.groupby('sid')['item_id'].apply(list).to_dict()

    return gt_df, sid_to_items

def parse_and_map(predict_list, sid_to_items_map):
    final_item_list = []
    total_sid_count = len(predict_list)
    parse_fail_count = 0
    map_fail_count = 0

    for sid_str in predict_list:
        match = SID_PATTERN.fullmatch(sid_str)

        if match:
            if sid_str in sid_to_items_map:
                item_candidates = sid_to_items_map[sid_str]
                num_to_sample = min(MAX_ITEM_PER_SID, len(item_candidates))

                if num_to_sample > 0:
                    sampled_items = np.random.choice(
                        item_candidates,
                        size=num_to_sample,
                        replace=False
                    ).tolist()
                    final_item_list.extend(sampled_items)
                else:
                    map_fail_count += 1
            else:
                map_fail_count += 1
        else:
            parse_fail_count += 1

    pre_dedup_len = len(final_item_list)
    seen = set()
    deduped_item_list = []
    for item in final_item_list:
        if item not in seen:
            seen.add(item)
            deduped_item_list.append(item)
    duplicate_count = pre_dedup_len - len(deduped_item_list)
    final_item_list = deduped_item_list

    final_item_list = final_item_list[:MAX_PREDICT_LEN]

    return final_item_list, parse_fail_count, map_fail_count, total_sid_count, duplicate_count



def process_jsonl_file(file_path, gt_df, sid_to_items_map):
    all_predictions_data = []
    total_lines = 0
    parsed_lines = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            try:
                record = json.loads(line)
                raw_predict_value = record.get('predict')
                predict_list = []

                if isinstance(raw_predict_value, list):
                    predict_list = raw_predict_value

                elif raw_predict_value is None or (np.isscalar(raw_predict_value) and pd.isna(raw_predict_value)):
                    predict_list = []

                elif isinstance(raw_predict_value, str):
                    try:
                        predict_list = eval(raw_predict_value)
                    except Exception as e:
                        print(f"    Warning: Eval failed on line {total_lines} in {file_path}. Value: {raw_predict_value[:50]}... Error: {e}")
                        predict_list = []

                if not isinstance(predict_list, list):
                     print(f"    Warning: Final predict value is not a list on line {total_lines}. Type: {type(predict_list)}. Setting to [].")
                     predict_list = []

                all_predictions_data.append(predict_list)
                parsed_lines += 1

            except json.JSONDecodeError as e:
                print(f"    Error: JSON decode failed on line {total_lines} in {file_path}. Skipping line. Error: {e}")
    print(f"    -> Successfully parsed {parsed_lines} / {total_lines} lines.")

    if parsed_lines != len(gt_df):
        print(f"    Warning: Parsed lines count ({parsed_lines}) does not match groundtruth ({len(gt_df)}). Skipping file.")
        return None, None

    total_parse_fail_count = 0
    total_map_fail_count = 0
    total_item_list_len = 0
    total_predict_count = parsed_lines
    total_original_sid_count = 0
    total_duplicate_count = 0

    hr_results = defaultdict(list)
    ndcg_results = defaultdict(list)

    for idx, predict_list in enumerate(all_predictions_data):
        true_item = gt_df.iloc[idx]['item_id']

        final_items, parse_fail, map_fail, total_sids, duplicate_count = parse_and_map(predict_list, sid_to_items_map)

        total_parse_fail_count += parse_fail
        total_map_fail_count += map_fail
        total_item_list_len += len(final_items)
        total_original_sid_count += total_sids
        total_duplicate_count += duplicate_count

        for k in K_VALUES:
            hr = calculate_hit_rate(final_items, true_item, k)
            ndcg = calculate_ndcg(final_items, true_item, k)

            hr_results[f'HR@{k}'].append(hr)
            ndcg_results[f'NDCG@{k}'].append(ndcg)

    metrics_data = {}
    for k in K_VALUES:
        metrics_data[f'HR@{k}'] = np.mean(hr_results[f'HR@{k}'])
        metrics_data[f'NDCG@{k}'] = np.mean(ndcg_results[f'NDCG@{k}'])
    metrics_series = pd.Series(metrics_data)

    successful_parse_count = total_original_sid_count - total_parse_fail_count
    stats_data = {
        'total_data_count': total_predict_count,
        'total_original_sid_count': total_original_sid_count,
        'avg_parse_fail_count': total_parse_fail_count / total_predict_count,
        'parse_fail_ratio': total_parse_fail_count / total_original_sid_count if total_original_sid_count > 0 else 0,
        'avg_map_fail_count': total_map_fail_count / total_predict_count,
        'map_fail_ratio': total_map_fail_count / successful_parse_count if successful_parse_count > 0 else 0,
        'avg_final_item_id_list_length': total_item_list_len / total_predict_count if total_predict_count > 0 else 0,
        'avg_item_id_duplicate_count': total_duplicate_count / total_predict_count if total_predict_count > 0 else 0
    }
    stats_series = pd.Series(stats_data)

    return metrics_series, stats_series

def main():
    if not os.path.isdir(JSONL_DIR):
        print(f"Error: JSONL directory not found at {JSONL_DIR}")
        return
    if not os.path.exists(GROUNDTRUTH_PATH):
        print(f"Error: Groundtruth CSV not found at {GROUNDTRUTH_PATH}")
        return
    if not os.path.exists(SID_ITEM_MAP_PATH):
        print(f"Error: SID-Item Map CSV not found at {SID_ITEM_MAP_PATH}")
        return

    print("Step 1: Loading Groundtruth and Mapping Data...")
    try:
        gt_df, sid_to_items_map = load_data()
    except Exception as e:
        print(f"Failed to load initial data: {e}")
        return
    print(f"Loaded {len(gt_df)} groundtruth records and {len(sid_to_items_map)} unique sids in map.")

    all_metrics = {}
    all_stats = {}

    jsonl_files = [f for f in os.listdir(JSONL_DIR) if f.endswith('.jsonl')]

    print(f"\nStep 2: Processing {len(jsonl_files)} JSONL files...")

    for filename in jsonl_files:
        file_path = os.path.join(JSONL_DIR, filename)
        file_name_without_ext = os.path.splitext(filename)[0]

        print(f"-> Processing {filename}...")

        metrics_series, stats_series = process_jsonl_file(file_path, gt_df, sid_to_items_map)

        if metrics_series is not None and stats_series is not None:
            all_metrics[file_name_without_ext] = metrics_series
            all_stats[file_name_without_ext] = stats_series

    if not all_metrics:
        print("\nNo files were successfully processed. Exiting.")
        return

    print("\nStep 3: Compiling and Saving Results...")

    CHECKPOINT_PATTERN = re.compile(r'(?:checkpoint-|ckpt_)(\d+)')

    def extract_checkpoint_number(filename):
        match = CHECKPOINT_PATTERN.search(filename)
        if match:
            return int(match.group(1))
        return float('inf')

    jsonl_names = list(all_metrics.keys())
    sorted_names = sorted(jsonl_names, key=extract_checkpoint_number)

    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.index.name = 'jsonl_name'
    metrics_df = metrics_df.reindex(sorted_names)

    cols_order = []
    for k in K_VALUES:
        cols_order.extend([f'HR@{k}', f'NDCG@{k}'])

    existing_cols = [col for col in cols_order if col in metrics_df.columns]
    if existing_cols:
        metrics_df = metrics_df[existing_cols]

    metrics_df.to_csv(OUTPUT_METRICS_PATH, float_format='%.6f')
    print(f"Metrics saved to {OUTPUT_METRICS_PATH} (Sorted by checkpoint).")

    stats_df = pd.DataFrame(all_stats).T
    stats_df.index.name = 'jsonl_name'
    stats_df = stats_df.reindex(sorted_names)

    stats_df.to_csv(OUTPUT_STATS_PATH, float_format='%.4f')
    print(f"Statistics saved to {OUTPUT_STATS_PATH} (Sorted by checkpoint).")

    print("\nDone!")

if __name__ == "__main__":
    main()
    


