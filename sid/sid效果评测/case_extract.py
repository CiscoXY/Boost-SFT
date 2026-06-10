"""
sid抽样分析
"""
import pandas as pd
import json
import random
import os
import argparse
import re
from collections import defaultdict
from pathlib import Path


def sanitize_filename(sid):
    """
    将 sid 转换为安全的文件名
    例如：<a_218><b_31><c_159> -> a_218_b_31_c_159
    """
    # 提取所有 <...> 中的内容，并用 _ 连接
    parts = re.findall(r'<([^>]+)>', sid)
    if parts:
        return '_'.join(parts)
    else:
        # 如果没有尖括号，移除文件系统非法字符
        return re.sub(r'[\\/:*?"<>|]', '_', sid)


def main(csv_path, jsonl_path, m, k, output_dir, seed=None, at_least_m=False):
    """
    主函数
    
    Args:
        csv_path: CSV 文件路径
        jsonl_path: JSONL 文件路径  
        m: item 数量阈值（恰好 m 个或至少 m 个）
        k: 随机抽取的 sid 数量
        output_dir: 输出目录
        seed: 随机种子（可选，保证可复现）
        at_least_m: 如果为 True，筛选"至少 m 个"的 sid；否则筛选"恰好 m 个"
    """
    # 设置随机种子（如果需要可复现）
    if seed is not None:
        random.seed(seed)
        print(f"设置随机种子: {seed}")
    
    # 1. 读取 CSV，建立 sid 到 item_id 列表的映射
    print(f"读取 CSV 文件: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 检查必要列
    if 'sid' not in df.columns or 'item_id' not in df.columns:
        raise ValueError("CSV 必须包含 'sid' 和 'item_id' 列")
    
    sid_to_items = defaultdict(list)
    for _, row in df.iterrows():
        sid_to_items[row['sid']].append(row['item_id'])
    
    print(f"共发现 {len(sid_to_items)} 个唯一的 sid")
    
    # 2. 筛选符合条件的 sid
    if at_least_m:
        eligible_sids = [sid for sid, items in sid_to_items.items() if len(items) >= m]
        mode_str = "至少"
    else:
        eligible_sids = [sid for sid, items in sid_to_items.items() if len(items) == m]
        mode_str = "恰好"
    
    print(f"筛选出 {mode_str} {m} 个 item 的 sid 数量: {len(eligible_sids)}")
    
    if len(eligible_sids) == 0:
        print(f"警告: 没有找到{mode_str} {m} 个 item 的 sid，程序退出")
        return
    
    if len(eligible_sids) < k:
        print(f"警告: 符合条件的 sid 数量 ({len(eligible_sids)}) 少于需要抽取的 k ({k})，将抽取全部")
        k = len(eligible_sids)
    
    # 3. 随机抽取 k 个 sid
    selected_sids = random.sample(eligible_sids, k)
    print(f"随机抽取了 {k} 个 sid: {selected_sids[:3]}{'...' if k > 3 else ''}")
    
    # 4. 读取 JSONL，建立 item_id 到 metadata 的映射
    print(f"读取 JSONL 文件: {jsonl_path}")
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
                print(f"警告: 第 {line_num} 行 JSON 解析错误: {e}")
                continue
    
    print(f"共加载 {len(item_metadata)} 条 metadata 记录")
    
    # 5. 为每个选中的 sid 生成 jsonl 文件
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    success_count = 0
    for sid in selected_sids:
        item_ids = sid_to_items[sid]
        safe_filename = sanitize_filename(sid)
        output_path = os.path.join(output_dir, f"{safe_filename}.jsonl")
        
        # 收集该 sid 下所有 item 的 metadata
        records = []
        for item_id in item_ids:
            if item_id in item_metadata:
                records.append(item_metadata[item_id])
            else:
                missing_items.add(item_id)
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"生成文件: {safe_filename}.jsonl (包含 {len(records)} 个 item)")
        success_count += 1
    
    if missing_items:
        print(f"警告: 有 {len(missing_items)} 个 item_id 在 metadata 中找不到对应记录")
    
    print(f"完成! 成功生成 {success_count} 个文件")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='从 CSV 中筛选指定 item 数量的 sid，随机抽取并输出对应的 metadata'
    )
    parser.add_argument('--csv', required=True, help='输入 CSV 文件路径')
    parser.add_argument('--jsonl', required=True, help='输入 meta_data.jsonl 文件路径')
    parser.add_argument('-m', type=int, required=True, help='筛选条件：item 数量')
    parser.add_argument('-k', type=int, required=True, help='随机抽取的 sid 数量')
    parser.add_argument('--output', default='./output_sids', help='输出目录（默认: ./output_sids）')
    parser.add_argument('--seed', type=int, default=None, help='随机种子（可选，用于结果复现）')
    parser.add_argument('--at-least', action='store_true', 
                       help='如果设置，筛选"至少 m 个"item 的 sid；否则筛选"恰好 m 个"')
    
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