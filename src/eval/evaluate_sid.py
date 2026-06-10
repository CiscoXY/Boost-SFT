import ast
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd


# GROUNDTRUTH_PATH = '/mnt/data/userseq/test/Books.csv'
# SID_ITEM_MAP_PATH = '/mnt/data/sid/Books/checkpoint_120000/itemid_to_sid.csv'
# JSONL_DIR = '/mnt/data/generate/Books/sft_loss_version_权重金龙鱼_消融_tp_06_n_20'

# GROUNDTRUTH_PATH = '/mnt/data/userseq/test/Beauty.csv'
# SID_ITEM_MAP_PATH = '/mnt/data/sid/Beauty/checkpoint_120000/itemid_to_sid.csv'
# JSONL_DIR = '/mnt/data/generate/Beauty/sft_loss_version_权重金龙鱼_消融_tp_06_n_20'


GROUNDTRUTH_PATH = '/mnt/data/userseq/test/Sports_Outdoors.csv'
SID_ITEM_MAP_PATH = '/mnt/data/sid/Sports_Outdoors/checkpoint_120000/itemid_to_sid.csv'
JSONL_DIR = "/mnt/data/generate/Sports_Outdoors/sft_loss_version_权重金龙鱼_消融_tp_06_n_20"

# 所有结果会统一放到这个子目录下。
OUTPUT_DIR = os.path.join(JSONL_DIR, 'sid_level_metrics')

K_VALUES = [1, 3, 5, 10, 20]
SID_LEVELS = {
    'sid1': 1,  # 只要求 <a_x> 命中
    'sid2': 2,  # 要求 <a_x><b_y> 命中
    'sid3': 3,  # 要求 <a_x><b_y><c_z> 全命中
}

# 形如 <a_10><b_113><c_64>
SID_PATTERN = re.compile(r'<a_(\d{1,3})><b_(\d{1,3})><c_(\d{1,3})>')
MAX_PREDICT_LEN = 20


def calculate_hit_rate(predicted_items, true_item, k):
    """计算 Hit Rate @k。"""
    if k <= 0:
        return np.nan

    pred_k = predicted_items[:k]
    return 1.0 if true_item in pred_k else 0.0


def calculate_ndcg(predicted_items, true_item, k):
    """计算单标签 NDCG @k。"""
    if k <= 0:
        return np.nan

    pred_k = predicted_items[:k]
    if true_item not in pred_k:
        return 0.0

    relevance = np.array([1 if item == true_item else 0 for item in pred_k])
    ranks = np.arange(1, len(relevance) + 1)
    denominator = np.log2(ranks + 1)
    dcg = np.sum(relevance / denominator)

    return dcg


def load_data():
    """加载 groundtruth 和 item_id -> sid 映射。"""
    gt_df = pd.read_csv(GROUNDTRUTH_PATH)
    if 'item_id' not in gt_df.columns:
        raise ValueError("Groundtruth CSV 必须包含 'item_id' 列。")

    map_df = pd.read_csv(SID_ITEM_MAP_PATH, dtype={'item_id': 'int', 'sid': 'str'})
    if 'item_id' not in map_df.columns or 'sid' not in map_df.columns:
        raise ValueError("SID-Item Map CSV 必须包含 'item_id' 和 'sid' 列。")

    map_df = map_df.dropna(subset=['item_id', 'sid']).drop_duplicates(subset=['item_id'], keep='first')
    item_to_sid = dict(zip(map_df['item_id'].astype(int), map_df['sid'].astype(str)))

    return gt_df, item_to_sid


def sid_to_level_key(sid, level):
    """
    把完整 SID 转成指定层级的比较 key：
    - level=1: <a_x>
    - level=2: <a_x><b_y>
    - level=3: <a_x><b_y><c_z>
    """
    if not isinstance(sid, str):
        return None

    match = SID_PATTERN.fullmatch(sid.strip())
    if not match:
        return None

    a_id, b_id, c_id = match.groups()
    if level == 1:
        return f'<a_{a_id}>'
    if level == 2:
        return f'<a_{a_id}><b_{b_id}>'
    if level == 3:
        return f'<a_{a_id}><b_{b_id}><c_{c_id}>'

    raise ValueError(f'Unsupported SID level: {level}')


def normalize_predict_list(raw_predict_value, line_num, file_path):
    """把 jsonl 中的 predict 字段统一解析成 list。"""
    if isinstance(raw_predict_value, list):
        return raw_predict_value

    if raw_predict_value is None or (np.isscalar(raw_predict_value) and pd.isna(raw_predict_value)):
        return []

    if isinstance(raw_predict_value, str):
        try:
            parsed = ast.literal_eval(raw_predict_value)
        except Exception as exc:
            print(
                f"    Warning: Parse predict failed on line {line_num} in {file_path}. "
                f"Value: {raw_predict_value[:50]}... Error: {exc}"
            )
            return []

        if isinstance(parsed, list):
            return parsed

    print(
        f"    Warning: Final predict value is not a list on line {line_num}. "
        f"Type: {type(raw_predict_value)}. Setting to []."
    )
    return []


def parse_sid_predictions(predict_list, level):
    """
    将预测的完整 SID list 转为指定层级的 SID key list。

    返回：
    - final_sid_list: 层级 SID 列表，去重后截断到 MAX_PREDICT_LEN
    - parse_fail_count: 不符合完整 SID 格式的预测数量
    - total_sid_count: 原始预测数量
    - duplicate_count: 转成层级 key 后的重复数量
    """
    level_sid_list = []
    total_sid_count = len(predict_list)
    parse_fail_count = 0

    for sid_str in predict_list:
        level_key = sid_to_level_key(sid_str, level)
        if level_key is None:
            parse_fail_count += 1
            continue

        level_sid_list.append(level_key)

    pre_dedup_len = len(level_sid_list)
    seen = set()
    deduped_sid_list = []
    for sid_key in level_sid_list:
        if sid_key not in seen:
            seen.add(sid_key)
            deduped_sid_list.append(sid_key)

    duplicate_count = pre_dedup_len - len(deduped_sid_list)
    final_sid_list = deduped_sid_list[:MAX_PREDICT_LEN]

    return final_sid_list, parse_fail_count, total_sid_count, duplicate_count


def process_jsonl_file(file_path, gt_df, item_to_sid_map, sid_level):
    """处理单个 jsonl 文件，计算指定 SID 粒度下的指标和统计信息。"""
    all_predictions_data = []
    total_lines = 0
    parsed_lines = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"    Error: JSON decode failed on line {total_lines} in {file_path}. Skipping line. Error: {exc}")
                continue

            predict_list = normalize_predict_list(record.get('predict'), total_lines, file_path)
            all_predictions_data.append(predict_list)
            parsed_lines += 1

    print(f"    -> Successfully parsed {parsed_lines} / {total_lines} lines.")

    if parsed_lines != len(gt_df):
        print(f"    Warning: Parsed lines count ({parsed_lines}) does not match groundtruth ({len(gt_df)}). Skipping file.")
        return None, None

    total_parse_fail_count = 0
    total_sid_list_len = 0
    total_predict_count = parsed_lines
    total_original_sid_count = 0
    total_duplicate_count = 0
    missing_truth_sid_count = 0

    hr_results = defaultdict(list)
    ndcg_results = defaultdict(list)

    for idx, predict_list in enumerate(all_predictions_data):
        true_item = int(gt_df.iloc[idx]['item_id'])
        true_sid = item_to_sid_map.get(true_item)
        true_sid_key = sid_to_level_key(true_sid, sid_level)

        if true_sid_key is None:
            missing_truth_sid_count += 1
            continue

        final_sids, parse_fail, total_sids, duplicate_count = parse_sid_predictions(predict_list, sid_level)

        total_parse_fail_count += parse_fail
        total_sid_list_len += len(final_sids)
        total_original_sid_count += total_sids
        total_duplicate_count += duplicate_count

        for k in K_VALUES:
            hr_results[f'HR@{k}'].append(calculate_hit_rate(final_sids, true_sid_key, k))
            ndcg_results[f'NDCG@{k}'].append(calculate_ndcg(final_sids, true_sid_key, k))

    evaluable_count = total_predict_count - missing_truth_sid_count
    metrics_data = {}
    for k in K_VALUES:
        metrics_data[f'HR@{k}'] = np.mean(hr_results[f'HR@{k}']) if hr_results[f'HR@{k}'] else np.nan
        metrics_data[f'NDCG@{k}'] = np.mean(ndcg_results[f'NDCG@{k}']) if ndcg_results[f'NDCG@{k}'] else np.nan
    metrics_series = pd.Series(metrics_data)

    stats_data = {
        '总数据量': total_predict_count,
        '可评估数据量': evaluable_count,
        '真实item缺失SID数量': missing_truth_sid_count,
        '原始预测SID总数量': total_original_sid_count,
        '平均解析失败个数': total_parse_fail_count / evaluable_count if evaluable_count > 0 else 0,
        '解析失败占比': total_parse_fail_count / total_original_sid_count if total_original_sid_count > 0 else 0,
        '平均最终SID列表长度': total_sid_list_len / evaluable_count if evaluable_count > 0 else 0,
        'SID重复数量均值': total_duplicate_count / evaluable_count if evaluable_count > 0 else 0,
    }
    stats_series = pd.Series(stats_data)

    return metrics_series, stats_series


def extract_checkpoint_number(filename):
    """从文件名/文件夹名中提取 checkpoint/ckpt 对应的数字，用于排序。"""
    checkpoint_pattern = re.compile(r'(?:checkpoint-|ckpt_)(\d+)')
    match = checkpoint_pattern.search(filename)
    if match:
        return int(match.group(1))
    return float('inf')


def save_level_results(output_dir, level_name, all_metrics, all_stats):
    """保存某个 SID 粒度下的 metrics 和 stats。"""
    if not all_metrics:
        print(f"    Warning: No files were successfully processed for {level_name}.")
        return

    sorted_names = sorted(all_metrics.keys(), key=extract_checkpoint_number)

    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.index.name = 'jsonl_name'
    metrics_df = metrics_df.reindex(sorted_names)

    cols_order = []
    for k in K_VALUES:
        cols_order.extend([f'HR@{k}', f'NDCG@{k}'])
    existing_cols = [col for col in cols_order if col in metrics_df.columns]
    if existing_cols:
        metrics_df = metrics_df[existing_cols]

    stats_df = pd.DataFrame(all_stats).T
    stats_df.index.name = 'jsonl_name'
    stats_df = stats_df.reindex(sorted_names)

    metrics_path = os.path.join(output_dir, f'{level_name}_metrics_summary.csv')
    stats_path = os.path.join(output_dir, f'{level_name}_stats_summary.csv')

    metrics_df.to_csv(metrics_path, float_format='%.6f')
    stats_df.to_csv(stats_path, float_format='%.4f')

    print(f"    Metrics saved to {metrics_path}")
    print(f"    Statistics saved to {stats_path}")


def main():
    """主函数，负责控制流程、文件加载和结果保存。"""
    if not os.path.isdir(JSONL_DIR):
        print(f"Error: JSONL directory not found at {JSONL_DIR}")
        return
    if not os.path.exists(GROUNDTRUTH_PATH):
        print(f"Error: Groundtruth CSV not found at {GROUNDTRUTH_PATH}")
        return
    if not os.path.exists(SID_ITEM_MAP_PATH):
        print(f"Error: SID-Item Map CSV not found at {SID_ITEM_MAP_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Step 1: Loading Groundtruth and Item->SID Mapping Data...")
    try:
        gt_df, item_to_sid_map = load_data()
    except Exception as exc:
        print(f"Failed to load initial data: {exc}")
        return
    print(f"Loaded {len(gt_df)} groundtruth records and {len(item_to_sid_map)} item->sid mappings.")

    jsonl_files = sorted([f for f in os.listdir(JSONL_DIR) if f.endswith('.jsonl')])
    print(f"\nStep 2: Processing {len(jsonl_files)} JSONL files for SID-level metrics...")

    results_by_level = {
        level_name: {
            'metrics': {},
            'stats': {},
        }
        for level_name in SID_LEVELS
    }

    for filename in jsonl_files:
        file_path = os.path.join(JSONL_DIR, filename)
        file_name_without_ext = os.path.splitext(filename)[0]
        print(f"-> Processing {filename}...")

        for level_name, sid_level in SID_LEVELS.items():
            print(f"    [{level_name}] evaluating...")
            metrics_series, stats_series = process_jsonl_file(file_path, gt_df, item_to_sid_map, sid_level)
            if metrics_series is None or stats_series is None:
                continue

            results_by_level[level_name]['metrics'][file_name_without_ext] = metrics_series
            results_by_level[level_name]['stats'][file_name_without_ext] = stats_series

    print(f"\nStep 3: Saving 6 result CSV files to {OUTPUT_DIR}...")
    for level_name in SID_LEVELS:
        save_level_results(
            OUTPUT_DIR,
            level_name,
            results_by_level[level_name]['metrics'],
            results_by_level[level_name]['stats'],
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
