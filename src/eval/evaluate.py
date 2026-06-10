import pandas as pd
import numpy as np
import re
import os
import json
from collections import defaultdict

GROUNDTRUTH_PATH = '/mnt/data/userseq/test/Books.csv'  
SID_ITEM_MAP_PATH = '/mnt/data/sid/Books/checkpoint_120000/itemid_to_sid.csv'  # sid <-> item_id 映射 CSV 路径

JSONL_DIR = '/mnt/data/generate/Books/sft_loss_version_tp_06_n_20'  # 存放所有 .jsonl 文件的目录
OUTPUT_METRICS_PATH = '/mnt/data/generate/Books/sft_loss_version_tp_06_n_20/metrics_summary.csv'  # 结果指标 CSV 路径
OUTPUT_STATS_PATH = '/mnt/data/generate/Books/sft_loss_version_tp_06_n_20/stats_summary.csv'  # 统计信息 CSV 路径

K_VALUES = [1, 3, 5, 10, 20]  # 计算指标的 k 值

# 正则表达式用于解析 sid: 形如 <a_10><b_113><c_64>
SID_PATTERN = re.compile(r'<a_(\d{1,3})><b_(\d{1,3})><c_(\d{1,3})>')
MAX_ITEM_PER_SID = 3 # 每个 sid 随机取的最大 item_id 数量
MAX_PREDICT_LEN = 20 # 最终预测列表的截断长度

# --- 1. 指标计算函数 ---

def calculate_hit_rate(predicted_items, true_item, k):
    """计算 Hit Rate @k"""
    if k <= 0:
        return np.nan
    # 截断到 k
    pred_k = predicted_items[:k]
    
    # 真实标签是一个 int 形式的 item_id，所以只需要检查是否在列表中
    return 1.0 if true_item in pred_k else 0.0

def calculate_ndcg(predicted_items, true_item, k):
    """计算 NDCG @k"""
    if k <= 0:
        return np.nan
    
    pred_k = predicted_items[:k]
    
    # 真实标签的 item_id 只有 1 个，我们假设是一个二元相关性 (1 if hit, 0 if not)
    relevance = np.array([1 if item == true_item else 0 for item in pred_k])
    
    # 检查是否有命中，如果未命中，NDCG 为 0
    if true_item not in pred_k:
        return 0.0
        
    # 计算 DCG
    # 索引从 0 开始，排名从 1 开始
    # DCG = sum(rel_i / log2(i + 1 + 1)) = sum(rel_i / log2(i + 2))
    # where i is 0-indexed position (0 to k-1)
    i_plus_1 = np.arange(1, len(relevance) + 1) # 1, 2, ..., k
    # 计算 log2(i+1) for the denominator (log2(rank+1))
    denominator = np.log2(i_plus_1 + 1) # log2(2), log2(3), ..., log2(k+1)
    
    dcg = np.sum(relevance / denominator)
    
    # 计算 IDCG (理想 DCG)
    # 对于单个真实标签，IDCG 总是 1 / log2(1 + 1) = 1.0
    idcg = 1.0 
    
    return dcg / idcg

# --- 2. 预处理函数 ---

def load_data():
    """加载 groundtruth 和 sid-item 映射数据"""
    
    # groundtruth
    # 确保 'item_id' 列是 int 类型
    gt_df = pd.read_csv(GROUNDTRUTH_PATH)
    if 'item_id' not in gt_df.columns:
        raise ValueError(f"Groundtruth CSV 必须包含 'item_id' 列。")
    # 假设 groundtruth 包含一个唯一标识符列用于和 jsonl 行对应
    
    # sid <-> item 映射
    # item_id: int, sid: str
    map_df = pd.read_csv(SID_ITEM_MAP_PATH, dtype={'item_id': 'int', 'sid': 'str'})
    
    # 建立 sid 到 item_id 列表的映射 (可能一个 sid 对应多个 item_id)
    sid_to_items = map_df.groupby('sid')['item_id'].apply(list).to_dict()
    
    return gt_df, sid_to_items

def parse_and_map(predict_list, sid_to_items_map):
    """
    对 predict list 中的每个 str 进行解析和映射。
    返回: 
    - final_item_list: 映射后的 item_id 列表 (已去重、截断)
    - parse_fail_count: 解析失败的 sid 数量
    - map_fail_count: 解析成功但映射失败的 sid 数量
    - total_sid_count: 原始预测列表长度
    - duplicate_count: item_id 列表去重前的重复数量 【新增】
    """
    final_item_list = []
    total_sid_count = len(predict_list)
    parse_fail_count = 0
    map_fail_count = 0
    
    for sid_str in predict_list:
        match = SID_PATTERN.fullmatch(sid_str)
        
        if match:
            # 构造标准化的 sid 字符串（如果需要，但这里直接用原始字符串）
            # 由于映射表是用 str 存储的，我们直接用原始 sid_str 作为 key
            
            # 映射回 item_id
            if sid_str in sid_to_items_map:
                item_candidates = sid_to_items_map[sid_str]
                
                # 随机取 1 到 MAX_ITEM_PER_SID 个 item_id
                num_to_sample = min(MAX_ITEM_PER_SID, len(item_candidates))
                
                # 确保 item_candidates 不是空列表
                if num_to_sample > 0:
                    # 使用 np.random.choice 进行随机采样，replace=False 表示不重复采样
                    sampled_items = np.random.choice(
                        item_candidates, 
                        size=num_to_sample, 
                        replace=False
                    ).tolist()
                    final_item_list.extend(sampled_items)
                else:
                    # 映射表中有 sid 键，但值列表为空 (理论上不应发生，但作为安全措施)
                    map_fail_count += 1
            else:
                # 解析成功，但映射失败 (sid 在映射 CSV 中不存在)
                map_fail_count += 1
        else:
            # 正则表达式解析失败
            parse_fail_count += 1
    
    pre_dedup_len = len(final_item_list)  # 去重前长度
    # 去重：保留第一次出现的item_id，保证预测顺序不变
    seen = set()
    deduped_item_list = []
    for item in final_item_list:
        if item not in seen:
            seen.add(item)
            deduped_item_list.append(item)
    # 计算重复数量（去重前长度 - 去重后长度）
    duplicate_count = pre_dedup_len - len(deduped_item_list)
    # 更新为去重后的列表
    final_item_list = deduped_item_list
    # ======================================================
    
    # 3. 截断到前 MAX_PREDICT_LEN 个
    final_item_list = final_item_list[:MAX_PREDICT_LEN]
    
    return final_item_list, parse_fail_count, map_fail_count, total_sid_count, duplicate_count



def process_jsonl_file(file_path, gt_df, sid_to_items_map):
    """处理单个 jsonl 文件，计算指标和统计信息"""
    all_predictions_data = [] 
    total_lines = 0
    parsed_lines = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            try:
                record = json.loads(line)
                raw_predict_value = record.get('predict')
                predict_list = [] # 默认空列表

                if isinstance(raw_predict_value, list):
                    predict_list = raw_predict_value
            
                elif raw_predict_value is None or (np.isscalar(raw_predict_value) and pd.isna(raw_predict_value)):
                    predict_list = [] # 缺失值或None，返回空列表

                elif isinstance(raw_predict_value, str):
                    try:
                        # 尝试将字符串形式的列表转换成真正的列表
                        predict_list = eval(raw_predict_value)
                    except Exception as e:
                        print(f"    Warning: Eval failed on line {total_lines} in {file_path}. Value: {raw_predict_value[:50]}... Error: {e}")
                        predict_list = []
                        
                if not isinstance(predict_list, list):
                     print(f"    Warning: Final predict value is not a list on line {total_lines}. Type: {type(predict_list)}. Setting to [].")
                     predict_list = []

                # --- 结束核心修改区 ---

                all_predictions_data.append(predict_list)
                parsed_lines += 1

            except json.JSONDecodeError as e:
                print(f"    Error: JSON decode failed on line {total_lines} in {file_path}. Skipping line. Error: {e}")        
    print(f"    -> Successfully parsed {parsed_lines} / {total_lines} lines.")

    # 检查数据对齐
    if parsed_lines != len(gt_df):
        print(f"    Warning: Parsed lines count ({parsed_lines}) does not match groundtruth ({len(gt_df)}). Skipping file.")
        return None, None
    
    # 循环处理每一行数据和计算指标
    
    # 初始化统计信息
    total_parse_fail_count = 0
    total_map_fail_count = 0
    total_item_list_len = 0
    total_predict_count = parsed_lines 
    total_original_sid_count = 0 
    total_duplicate_count = 0  # 累计所有行的item_id重复数量
    
    # 初始化指标结果存储
    hr_results = defaultdict(list)
    ndcg_results = defaultdict(list)
    
    for idx, predict_list in enumerate(all_predictions_data):
        true_item = gt_df.iloc[idx]['item_id']
        
        # a. 解析和映射（接收新增的重复数量返回值）
        final_items, parse_fail, map_fail, total_sids, duplicate_count = parse_and_map(predict_list, sid_to_items_map)
        
        # b. 更新统计信息
        total_parse_fail_count += parse_fail
        total_map_fail_count += map_fail
        total_item_list_len += len(final_items)
        total_original_sid_count += total_sids
        total_duplicate_count += duplicate_count  # 【新增】累计重复数量
        
        # c. 计算指标 (使用原有逻辑)
        for k in K_VALUES:
            hr = calculate_hit_rate(final_items, true_item, k)
            ndcg = calculate_ndcg(final_items, true_item, k)
            
            hr_results[f'HR@{k}'].append(hr)
            ndcg_results[f'NDCG@{k}'].append(ndcg)
            
    # 4. 汇总计算平均指标和统计信息
    metrics_data = {}
    for k in K_VALUES:
        metrics_data[f'HR@{k}'] = np.mean(hr_results[f'HR@{k}'])
        metrics_data[f'NDCG@{k}'] = np.mean(ndcg_results[f'NDCG@{k}'])
    metrics_series = pd.Series(metrics_data)

    successful_parse_count = total_original_sid_count - total_parse_fail_count
    stats_data = {
        '总数据量': total_predict_count,
        '原始 SID 总数量': total_original_sid_count,
        '平均解析失败个数': total_parse_fail_count / total_predict_count,
        '解析失败占比': total_parse_fail_count / total_original_sid_count if total_original_sid_count > 0 else 0,
        '平均映射失败个数': total_map_fail_count / total_predict_count,
        '映射失败占比': total_map_fail_count / successful_parse_count if successful_parse_count > 0 else 0,
        '平均最终 Item ID 列表长度': total_item_list_len / total_predict_count if total_predict_count > 0 else 0,
        'item_id重复数量均值': total_duplicate_count / total_predict_count if total_predict_count > 0 else 0  # 【新增】重复数量均值
    }
    stats_series = pd.Series(stats_data)
    
    return metrics_series, stats_series

def main():
    """主函数，负责控制流程、文件加载和结果保存"""
    
    # 检查输入文件和目录是否存在
    if not os.path.isdir(JSONL_DIR):
        print(f"Error: JSONL directory not found at {JSONL_DIR}")
        return
    if not os.path.exists(GROUNDTRUTH_PATH):
        print(f"Error: Groundtruth CSV not found at {GROUNDTRUTH_PATH}")
        return
    if not os.path.exists(SID_ITEM_MAP_PATH):
        print(f"Error: SID-Item Map CSV not found at {SID_ITEM_MAP_PATH}")
        return
    
    print("✨ Step 1: Loading Groundtruth and Mapping Data...")
    try:
        gt_df, sid_to_items_map = load_data()
    except Exception as e:
        print(f"Failed to load initial data: {e}")
        return
    print(f"Loaded {len(gt_df)} groundtruth records and {len(sid_to_items_map)} unique sids in map.")
    
    all_metrics = {}
    all_stats = {}
    
    jsonl_files = [f for f in os.listdir(JSONL_DIR) if f.endswith('.jsonl')]
    
    print(f"\n✨ Step 2: Processing {len(jsonl_files)} JSONL files...")
    
    for filename in jsonl_files:
        file_path = os.path.join(JSONL_DIR, filename)
        file_name_without_ext = os.path.splitext(filename)[0]
        
        print(f"-> Processing {filename}...")
        
        metrics_series, stats_series = process_jsonl_file(file_path, gt_df, sid_to_items_map)
        
        if metrics_series is not None and stats_series is not None:
            all_metrics[file_name_without_ext] = metrics_series
            all_stats[file_name_without_ext] = stats_series
    
    if not all_metrics:
        print("\n❌ No files were successfully processed. Exiting.")
        return
        
    # 4. 结果输出
    print("\n✨ Step 3: Compiling and Saving Results...")
    
    CHECKPOINT_PATTERN = re.compile(r'(?:checkpoint-|ckpt_)(\d+)')

    def extract_checkpoint_number(filename):
        """从文件名/文件夹名中提取 checkpoint/ckpt 对应的数字，用于排序
        支持两种格式：
        1. checkpoint-N(如 checkpoint-100、abc_checkpoint-200_def)
        2. ckpt_N(如 model_ckpt_300、ckpt_400_folder)
        """
        match = CHECKPOINT_PATTERN.search(filename)
        if match:
            # 转换为整数进行升序排序
            return int(match.group(1))
        # 对于不匹配两种格式的名称，返回无穷大，确保排在最后
        return float('inf')

    jsonl_names = list(all_metrics.keys()) # 文件名 (不含后缀)
    sorted_names = sorted(jsonl_names, key=extract_checkpoint_number)
    
    # -----------------------------------------------------
    # 结果输出
    
    # 第一个 CSV: 指标
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.index.name = 'jsonl_name'
    
    # 重新索引 DataFrame 以实现排序
    metrics_df = metrics_df.reindex(sorted_names) 
    
    # 确保列顺序是 HR@1, NDCG@1, ... (与原代码保持一致)
    cols_order = []
    for k in K_VALUES:
        cols_order.extend([f'HR@{k}', f'NDCG@{k}'])
    
    # 检查所有目标列是否在 DataFrame 中
    existing_cols = [col for col in cols_order if col in metrics_df.columns]
    if existing_cols:
        metrics_df = metrics_df[existing_cols]

    metrics_df.to_csv(OUTPUT_METRICS_PATH, float_format='%.6f')
    print(f"✅ Metrics saved to {OUTPUT_METRICS_PATH} (Sorted by checkpoint).")
    
    # 第二个 CSV: 统计信息
    stats_df = pd.DataFrame(all_stats).T
    stats_df.index.name = 'jsonl_name'
    
    # 重新索引 DataFrame 以实现排序
    stats_df = stats_df.reindex(sorted_names) 
    
    stats_df.to_csv(OUTPUT_STATS_PATH, float_format='%.4f')
    print(f"✅ Statistics saved to {OUTPUT_STATS_PATH} (Sorted by checkpoint).")
    
    print("\n🎉 Done!")

if __name__ == "__main__":
    main()
    


