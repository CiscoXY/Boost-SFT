import pandas as pd
import os
import json
import logging
from typing import Dict, List
import random

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
    df_mapping = df_mapping.dropna(subset=["item_id"]).drop_duplicates(subset=["item_id"], keep="first")
    df_mapping["item_id"] = df_mapping["item_id"].astype(int)
    
    mapping_dict = df_mapping.set_index("item_id")["sid"].to_dict()
    logging.info(f"Successfully built mapping dictionary with {len(mapping_dict)} valid item_id->sid mappings")
    return mapping_dict

def parse_truncate_config(truncate_config: Dict[str, int]) -> Dict[str, int]:
    """Validate and process field truncation configuration."""
    valid_truncate = {}
    for field, num in truncate_config.items():
        try:
            truncate_num = int(num)
            if truncate_num < 0:
                logging.warning(f"Truncation count for field {field} cannot be negative, set to 0 (no truncation)")
                truncate_num = 0
            valid_truncate[field] = truncate_num
        except ValueError:
            logging.warning(f"Truncation count for field {field} must be an integer, skipping this config")
    return valid_truncate

def load_and_process_item_meta(
    meta_path: str,
    mapping_dict: Dict[int, str],
    target_fields: List[str],
    truncate_dict: Dict[str, int],
    sample_ratio: float = 1.0
) -> List[Dict[str, str]]:
    """Load item_meta.jsonl, process field truncation, filter invalid items, and sample."""
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"item_meta.jsonl not found: {meta_path}")
    
    if not (0 < sample_ratio <= 1):
        raise ValueError(f"Sampling ratio must be in (0,1], current: {sample_ratio}")
    
    logging.info(f"Starting to load item_meta.jsonl, target fields: {target_fields}, sampling ratio: {sample_ratio}")
    
    processed_items = []
    total_count = 0
    valid_count = 0
    missing_field_count = 0
    missing_sid_count = 0
    
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_count += 1
            
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                logging.debug(f"Invalid JSON format (skipped): {line[:50]}...")
                continue
            
            item_id = item.get("item_id")
            if item_id is None:
                continue
            try:
                item_id = int(item_id)
            except (ValueError, TypeError):
                continue
            
            sid = mapping_dict.get(item_id)
            if sid is None:
                missing_sid_count += 1
                continue
            
            field_infos = []
            has_missing_field = False
            for field in target_fields:
                field_value = str(item.get(field, "")).strip()
                if not field_value:
                    has_missing_field = True
                    field_text = f"{field}:None"
                else:
                    truncate_num = truncate_dict.get(field, 0)
                    if truncate_num > 0:
                        words = field_value.split()
                        truncated_words = words[:truncate_num]
                        field_value = " ".join(truncated_words)
                    field_text = f"{field}:{field_value}"
                field_infos.append(field_text)
            
            item_info = ", ".join(field_infos) + " "
            
            if has_missing_field:
                missing_field_count += 1
            
            processed_items.append({
                "item_info": item_info,
                "sid": sid,
                "item_id": item_id
            })
            valid_count += 1
    
    if sample_ratio < 1.0:
        sample_size = int(len(processed_items) * sample_ratio)
        random.seed(42)
        processed_items = random.sample(processed_items, sample_size)
        logging.info(f"Sampling complete: sampled {sample_size} from {valid_count} valid items")
    else:
        logging.info(f"No sampling: {valid_count} valid items total")
    
    logging.info("item_meta processing statistics:")
    logging.info(f"  - Total records: {total_count}")
    logging.info(f"  - Valid records (with item_id+sid): {valid_count}")
    logging.info(f"  - Records without corresponding sid: {missing_sid_count}")
    logging.info(f"  - Valid records with missing fields: {missing_field_count}")
    logging.info(f"  - Final processed items: {len(processed_items)}")
    
    return processed_items

def construct_alignment_data(
    processed_items: List[Dict[str, str]],
    instruction1: str = "Generate the corresponding sid based on the item metadata",
    instruction2: str = "Generate the corresponding item metadata based on the sid"
) -> List[Dict[str, str]]:
    """Construct bidirectional alignment data: 2 records per item (item_info->sid and sid->item_info)."""
    logging.info("Starting to construct bidirectional alignment data...")
    
    alignment_data = []
    for item in processed_items:
        item_info = item["item_info"]
        sid = item["sid"]
        
        data1 = {
            "instruction": instruction1,
            "input": item_info,
            "output": sid
        }
        alignment_data.append(data1)
        
        data2 = {
            "instruction": instruction2,
            "input": f"sid:{sid}. ",
            "output": item_info
        }
        alignment_data.append(data2)
    
    logging.info(f"Data construction complete: generated {len(alignment_data)} records ({len(processed_items)} items × 2)")
    return alignment_data

def save_jsonl(data: List[Dict[str, str]], output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for line in data:
            json_line = json.dumps(line, ensure_ascii=False)
            f.write(json_line + "\n")
    
    logging.info(f"Successfully saved JSONL file to: {output_path}")
    logging.info(f"File size: {len(data)} records")

def main(config: Dict):
    try:
        logging.info("="*60)
        logging.info("Starting text alignment data processing pipeline")
        logging.info("="*60)
        
        mapping_dict = load_mapping_dict(config["mapping_csv"])
        
        logging.info("\n" + "="*60)
        truncate_dict = parse_truncate_config(config["truncate_config"])
        logging.info(f"Field truncation config: {truncate_dict if truncate_dict else 'None'}")
        
        logging.info("\n" + "="*60)
        processed_items = load_and_process_item_meta(
            meta_path=config["item_meta_path"],
            mapping_dict=mapping_dict,
            target_fields=config["target_fields"],
            truncate_dict=truncate_dict,
            sample_ratio=config["sample_ratio"]
        )
        
        logging.info("\n" + "="*60)
        alignment_data = construct_alignment_data(
            processed_items=processed_items,
            instruction1=config["instruction1"],
            instruction2=config["instruction2"]
        )
        
        logging.info("\n" + "="*60)
        save_jsonl(alignment_data, config["output_jsonl"])
        
        logging.info("\n" + "="*60)
        logging.info("All processes completed!")
        logging.info(f"Final output: {config['output_jsonl']} ({len(alignment_data)} bidirectional alignment records)")
        
    except Exception as e:
        logging.error(f"\nPipeline execution failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    CONFIG = {
        "mapping_csv": "/path/to/sid/checkpoint/itemid_to_sid.csv",
        "item_meta_path": "/path/to/dataset_items.jsonl",
        "output_jsonl": "/path/to/sid_understand/sid_understand_dataset.jsonl",
        
        "target_fields": ["title","features", "description"],
        "truncate_config": {
            "title": 0,
            "description": 300,
            "features": 150,
        },
        "sample_ratio": 0.5,
        
        "instruction1": "Please generate the corresponding semantic id based on the given item information",
        "instruction2": "Please restore the item information based on the given semantic id."
    }
    
    main(CONFIG)