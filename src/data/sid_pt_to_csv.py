import torch
import pandas as pd
from typing import Dict, List
import os

def load_pt_file(pt_path: str) -> Dict[str, List[str]]:
    try:
        data = torch.load(pt_path, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"Failed to load .pt file: {e}")
    
    if "item_to_tokens" not in data:
        raise KeyError("item_to_tokens field not found in .pt file")
    
    item_to_tokens = data["item_to_tokens"]
    
    for item_id, tokens in item_to_tokens.items():
        if not isinstance(tokens, list) or len(tokens) != 3:
            raise ValueError(f"item_id {item_id} has invalid tokens format, expected a list of length 3")
        for token in tokens:
            if not isinstance(token, str):
                raise ValueError(f"item_id {item_id} tokens contain non-string element: {token}")
    
    print(f"Successfully loaded item_to_tokens with {len(item_to_tokens)} items")
    return item_to_tokens

def convert_to_csv(item_to_tokens: Dict[str, List[str]], csv_path: str) -> None:
    csv_data = []
    for item_id, tokens in item_to_tokens.items():
        concat_token = ''.join(tokens)
        row = [item_id, concat_token] + tokens
        csv_data.append(row)
    
    columns = ["item_id", "sid", "sid1", "sid2", "sid3"]
    df = pd.DataFrame(csv_data, columns=columns)
    
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Successfully saved CSV file to: {csv_path}")
    print(f"CSV file has {len(df)} rows, {len(df.columns)} columns")
    return df

def generate_statistics_csv(df: pd.DataFrame, stat_csv_path: str) -> None:
    sid_item_count = df['sid'].value_counts().reset_index()
    sid_item_count.columns = ['sid', 'item_count']
    
    count_distribution = sid_item_count['item_count'].value_counts().reset_index()
    count_distribution.columns = ['item_count', 'sid_count']
    
    count_distribution = count_distribution.sort_values('item_count').reset_index(drop=True)
    
    total_sid = count_distribution['sid_count'].sum()
    count_distribution['cumulative_percent'] = (count_distribution['sid_count'].cumsum() / total_sid * 100).round(2)
    
    count_distribution.to_csv(stat_csv_path, index=False, encoding="utf-8")
    print(f"\nSuccessfully saved statistics CSV file to: {stat_csv_path}")
    print(f"Statistics CSV file has {len(count_distribution)} rows")
    print(f"Total unique sids: {total_sid}")
    print(f"Item count distribution range: {count_distribution['item_count'].min()} ~ {count_distribution['item_count'].max()}")

def main(sid_pt_path : str , output_dir : str):
    
    if not os.path.exists(sid_pt_path):
        raise FileNotFoundError(f"Input .pt file not found: {sid_pt_path}")
    
    if  not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    output_csv_path = os.path.join(output_dir , "itemid_to_sid.csv")
    output_stats_path = os.path.join(output_dir , "sid_statistics.csv")
    
    try:
        print("="*50)
        print("Starting to load .pt file...")
        item_to_tokens = load_pt_file(sid_pt_path)
        
        print("\n" + "="*50)
        print("Starting to convert to sid.csv file...")
        df = convert_to_csv(item_to_tokens, output_csv_path)
        
        print("\n" + "="*50)
        print("Starting to generate statistics CSV file...")
        generate_statistics_csv(df, output_stats_path)
        print("Conversion complete!")
    except Exception as e:
        print(f"Conversion failed: {e}")
        raise

if __name__ == "__main__":
    sid_path = "/path/to/sid/checkpoint/saved_tokenizer_data0.pt"
    output_dir = "/path/to/sid/checkpoint"
    main(sid_path , output_dir)