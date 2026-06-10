import pandas as pd
import argparse
import os
import ast
from typing import Dict, List, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def load_mapping_dict(mapping_csv_path: str) -> Dict[int, str]:
    """Load mapping CSV and build item_id -> sid mapping dictionary."""
    try:
        df_mapping = pd.read_csv(mapping_csv_path, encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to load mapping CSV: {e}")
    
    required_columns = ["item_id", "sid"]
    missing_cols = [col for col in required_columns if col not in df_mapping.columns]
    if missing_cols:
        raise KeyError(f"Mapping CSV missing required columns: {missing_cols}")
    
    df_mapping["item_id"] = pd.to_numeric(df_mapping["item_id"], errors="coerce")
    if df_mapping["item_id"].isna().any():
        logging.warning(f"{df_mapping['item_id'].isna().sum()} rows in mapping CSV have non-convertible item_id, skipped")
        df_mapping = df_mapping.dropna(subset=["item_id"])
    
    df_mapping["item_id"] = df_mapping["item_id"].astype(int)
    df_mapping = df_mapping.drop_duplicates(subset=["item_id"], keep="first")
    
    mapping_dict = df_mapping.set_index("item_id")["sid"].to_dict()
    logging.info(f"Successfully built mapping dictionary with {len(mapping_dict)} valid item_id->sid mappings")
    
    return mapping_dict

def load_userseq_csv(userseq_csv_path: str) -> pd.DataFrame:
    """Load userseq.csv and parse history_item_id field."""
    try:
        df_userseq = pd.read_csv(userseq_csv_path, encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to load userseq.csv: {e}")
    
    required_columns = ["history_item_id", "item_id"]
    missing_cols = [col for col in required_columns if col not in df_userseq.columns]
    if missing_cols:
        raise KeyError(f"userseq.csv missing required columns: {missing_cols}")
    
    def parse_history(history_str: str) -> Optional[List[int]]:
        try:
            history_list = ast.literal_eval(history_str)
            if not isinstance(history_list, list):
                return None
            history_int = [int(x) for x in history_list if isinstance(x, (int, float)) and pd.notna(x)]
            return history_int if history_int else None
        except:
            return None
    
    logging.info("Starting to parse history_item_id field...")
    df_userseq["history_item_list"] = df_userseq["history_item_id"].apply(parse_history)
    
    before_filter = len(df_userseq)
    df_userseq = df_userseq.dropna(subset=["history_item_list"])
    df_userseq = df_userseq[df_userseq["history_item_list"].apply(lambda x: len(x) > 0)]
    
    df_userseq["item_id"] = pd.to_numeric(df_userseq["item_id"], errors="coerce")
    df_userseq = df_userseq.dropna(subset=["item_id"])
    df_userseq["item_id"] = df_userseq["item_id"].astype(int)
    
    after_filter = len(df_userseq)
    logging.info(f"userseq.csv parsing complete: {before_filter} original rows, {after_filter} valid rows (filtered out failed parsing/empty history/invalid item_id)")
    
    return df_userseq[["history_item_list", "item_id"]]

def convert_userseq_data(
    df_userseq: pd.DataFrame,
    mapping_dict: Dict[int, str],
    missing_id_handling: str = "skip"
) -> pd.DataFrame:
    """Convert int-type ids in userseq to sid."""
    logging.info("Starting id->sid mapping conversion...")
    
    def map_history_to_sid(history_list: List[int]) -> Optional[str]:
        sid_list = [mapping_dict.get(item_id) for item_id in history_list]
        valid_sid_list = [sid for sid in sid_list if sid is not None]
        
        if missing_id_handling == "error" and len(valid_sid_list) != len(history_list):
            raise ValueError(f"Unmapped ids found in history item_id: {[item_id for item_id, sid in zip(history_list, sid_list) if sid is None]}")
        
        return ",".join(valid_sid_list) if valid_sid_list else None
    
    df_userseq["history_sid_str"] = df_userseq["history_item_list"].apply(map_history_to_sid)
    df_userseq["output_sid"] = df_userseq["item_id"].map(mapping_dict)
    
    before_filter = len(df_userseq)
    df_userseq = df_userseq.dropna(subset=["history_sid_str", "output_sid"])
    after_filter = len(df_userseq)
    
    logging.info(f"Mapping conversion complete: {before_filter} original rows, {after_filter} valid rows (filtered out rows without valid mappings)")
    
    return df_userseq[["history_sid_str", "output_sid"]]

def save_converted_csv(df_converted: pd.DataFrame, converted_csv_path: str) -> None:
    output_dir = os.path.dirname(converted_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    df_converted.to_csv(converted_csv_path, index=False, encoding="utf-8")
    logging.info(f"Successfully saved converted data to: {converted_csv_path}")
    logging.info(f"Converted data has {len(df_converted)} rows, columns: {list(df_converted.columns)}")

def convert_to_jsonl(
    df_converted: pd.DataFrame,
    jsonl_path: str,
    instruction: str,
    mode: str = "train"
) -> None:
    """Convert transformed data to JSONL format. In test mode, generates a separate groundtruth file."""
    logging.info("Starting JSONL format conversion...")
    
    output_dir = os.path.dirname(jsonl_path)
    jsonl_basename = os.path.splitext(os.path.basename(jsonl_path))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    if mode == "train":
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for _, row in df_converted.iterrows():
                json_line = {
                    "instruction": instruction,
                    "input": row["history_sid_str"],
                    "output": row["output_sid"]
                }
                import json
                f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
    elif mode == "test":
        ground_truth_path = os.path.join(output_dir , jsonl_basename + "_groundtruth.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for _, row in df_converted.iterrows():
                json_line = {
                    "instruction": instruction,
                    "input": row["history_sid_str"],
                }
                import json
                f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
        with open(ground_truth_path, "w", encoding="utf-8") as f:
            for _, row in df_converted.iterrows():
                json_line = {
                    "instruction": instruction,
                    "input": row["history_sid_str"],
                    "output": row["output_sid"]
                }
                import json
                f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
    
    logging.info(f"Successfully saved JSONL file to: {jsonl_path}")
    logging.info(f"JSONL file has {len(df_converted)} records")

def main():
    parser = argparse.ArgumentParser(description="Process userseq data: id->sid mapping conversion and generate JSONL training data")
    parser.add_argument("--mapping_csv", default="/path/to/sid/checkpoint/itemid_to_sid.csv", help="Path to the mapping CSV from the previous step (contains item_id and sid columns)")
    parser.add_argument("--userseq_csv", default="/path/to/userseq/valid/dataset.csv", help="Path to the input userseq.csv")
    parser.add_argument("--converted_csv_path", default="/path/to/userseq/valid/dataset_temp.csv", help="Path to save intermediate converted data (CSV format)")
    parser.add_argument("--jsonl_path", default="/path/to/userseq/valid/dataset_llm.jsonl", help="Path to the output JSONL file")
    parser.add_argument("--mode", default="train", help="Mode for generating user behavior data: 'train' or 'test'")
    parser.add_argument("--instruction", default="You are a professional search recommendation engine. Please predict the semantic id of the next item to be visited based on the user's historical visit records. The following are the historical records:", help="Instruction field content in JSONL")
    parser.add_argument(
        "--missing_id_handling",
        choices=["skip", "error"],
        default="error",
        help="How to handle unmapped item_ids: skip=skip the row, error=raise exception (default: error)"
    )
    
    args = parser.parse_args()
    
    for path in [args.mapping_csv, args.userseq_csv]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")
    
    try:
        logging.info("="*60)
        logging.info("Starting to load mapping dictionary...")
        mapping_dict = load_mapping_dict(args.mapping_csv)
        
        logging.info("\n" + "="*60)
        logging.info("Starting to load and parse userseq.csv...")
        df_userseq = load_userseq_csv(args.userseq_csv)
        
        logging.info("\n" + "="*60)
        df_converted = convert_userseq_data(
            df_userseq=df_userseq,
            mapping_dict=mapping_dict,
            missing_id_handling=args.missing_id_handling
        )
        
        logging.info("\n" + "="*60)
        save_converted_csv(df_converted, args.converted_csv_path)
        
        logging.info("\n" + "="*60)
        convert_to_jsonl(df_converted, args.jsonl_path, args.instruction , mode = args.mode)
        
        logging.info("\n" + "="*60)
        logging.info("All processes completed!")
        logging.info("Final output:")
        logging.info(f"  - Intermediate converted data: {args.converted_csv_path} ({len(df_converted)} rows)")
        logging.info(f"  - JSONL training data: {args.jsonl_path} ({len(df_converted)} records)")
    except Exception as e:
        logging.error(f"\nPipeline execution failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()