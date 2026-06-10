import torch
import pandas as pd
from typing import Dict, List, Tuple
import os

def load_pt_file(pt_path: str) -> Dict[str, List[str]]:
    """
    加载.pt文件并提取item_to_tokens部分
    
    Args:
        pt_path: .pt文件路径
        
    Returns:
        item_to_tokens字典，key为item_id，value为长度3的str列表
    """
    try:
        data = torch.load(pt_path, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"加载.pt文件失败: {e}")
    
    if "item_to_tokens" not in data:
        raise KeyError(".pt文件中未找到item_to_tokens字段")
    
    item_to_tokens = data["item_to_tokens"]
    
    # 验证数据格式
    for item_id, tokens in item_to_tokens.items():
        if not isinstance(tokens, list) or len(tokens) != 3:
            raise ValueError(f"item_id {item_id} 对应的tokens格式错误，应为长度3的列表")
        for token in tokens:
            if not isinstance(token, str):
                raise ValueError(f"item_id {item_id} 对应的tokens包含非字符串元素: {token}")
    
    print(f"成功加载item_to_tokens，共包含 {len(item_to_tokens)} 个item")
    return item_to_tokens

def convert_to_csv(item_to_tokens: Dict[str, List[str]], csv_path: str) -> None:
    """
    将item_to_tokens转换为指定格式的csv文件
    
    Args:
        item_to_tokens: 输入字典
        csv_path: 输出csv文件路径
    """
    csv_data = []
    for item_id, tokens in item_to_tokens.items():
        concat_token = ''.join(tokens)
        # 构建一行数据：item_id, concat_token, token1, token2, token3
        row = [item_id, concat_token] + tokens
        csv_data.append(row)
    
    columns = ["item_id", "sid", "sid1", "sid2", "sid3"]
    df = pd.DataFrame(csv_data, columns=columns)
    
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"成功保存csv文件到: {csv_path}")
    print(f"csv文件共 {len(df)} 行数据，{len(df.columns)} 列")
    return df

def generate_statistics_csv(df: pd.DataFrame, stat_csv_path: str) -> None:
    """
    生成统计信息csv：sid对应的item个数分布、频次、累计百分比
    
    Args:
        df: 主csv对应的DataFrame
        stat_csv_path: 统计csv输出路径
    """
    # 每个sid对应的item个数（sid出现次数）
    sid_item_count = df['sid'].value_counts().reset_index()
    sid_item_count.columns = ['sid', 'item_count']
    
    # 统计每个item_count对应的sid个数（即：有多少个sid对应这个item数量）
    count_distribution = sid_item_count['item_count'].value_counts().reset_index()
    count_distribution.columns = ['item_count', 'sid_count']
    
    count_distribution = count_distribution.sort_values('item_count').reset_index(drop=True)
    
    total_sid = count_distribution['sid_count'].sum()
    count_distribution['cumulative_percent'] = (count_distribution['sid_count'].cumsum() / total_sid * 100).round(2)
    
    count_distribution.to_csv(stat_csv_path, index=False, encoding="utf-8")
    print(f"\n成功保存统计csv文件到: {stat_csv_path}")
    print(f"统计csv文件共 {len(count_distribution)} 行数据")
    print(f"总共有 {total_sid} 个不同的sid")
    print(f"item个数分布范围：{count_distribution['item_count'].min()} ~ {count_distribution['item_count'].max()}")

def main(sid_pt_path : str , output_dir : str):
    
    if not os.path.exists(sid_pt_path):
        raise FileNotFoundError(f"输入.pt文件不存在: {sid_pt_path}")
    
    if  not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    output_csv_path = os.path.join(output_dir , "itemid_to_sid.csv")
    output_stats_path = os.path.join(output_dir , "sid_statistics.csv")
    
    try:
        print("="*50)
        print("开始加载.pt文件...")
        item_to_tokens = load_pt_file(sid_pt_path)
        
        print("\n" + "="*50)
        print("开始转换为sid.csv文件...")
        df = convert_to_csv(item_to_tokens, output_csv_path)
        
        print("\n" + "="*50)
        print("开始生成统计csv文件...")
        generate_statistics_csv(df, output_stats_path)
        print("转换完成！")
    except Exception as e:
        print(f"转换失败: {e}")
        raise

if __name__ == "__main__":
    sid_path = "/mnt/data/sid/Toys_and_Games/checkpoint_120000/saved_tokenizer_data0.pt"
    output_dir = "/mnt/data/sid/Toys_and_Games/checkpoint_120000"
    main(sid_path , output_dir)