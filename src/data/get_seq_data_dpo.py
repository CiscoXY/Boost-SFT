import pandas as pd
import argparse
import os
import ast
import json
from typing import Dict, List, Optional, Tuple
import logging
import random
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def load_mapping_dict(mapping_csv_path: str) -> Tuple[Dict[int, str], Dict[int, Tuple[str, str, str]], List[Tuple[str, str, str, str]], List[str], List[str], List[str], set[str]]:
    """Load mapping CSV and build item_id->sid, item_id->(sid1,sid2,sid3) mappings, plus sid pools."""
    try:
        df_mapping = pd.read_csv(mapping_csv_path, encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to load mapping CSV: {e}")
    
    required_columns = ["item_id", "sid", "sid1", "sid2", "sid3"]
    missing_cols = [col for col in required_columns if col not in df_mapping.columns]
    if missing_cols:
        raise KeyError(f"Mapping CSV missing required columns: {missing_cols}")
    
    df_mapping["item_id"] = pd.to_numeric(df_mapping["item_id"], errors="coerce")
    if df_mapping["item_id"].isna().any():
        logging.warning(f"{df_mapping['item_id'].isna().sum()} rows in mapping CSV have non-convertible item_id, skipped")
        df_mapping = df_mapping.dropna(subset=["item_id"])
    
    sid_cols = ["sid1", "sid2", "sid3"]
    for col in sid_cols:
        df_mapping[col] = df_mapping[col].astype(str).str.strip()
        empty_count = df_mapping[col].isin(["", "nan", None]).sum()
        if empty_count > 0:
            logging.warning(f"Column {col} in mapping CSV has {empty_count} empty rows, skipped")
            df_mapping = df_mapping[df_mapping[col].notna() & (df_mapping[col] != "")]
    
    df_mapping["item_id"] = df_mapping["item_id"].astype(int)
    df_mapping = df_mapping.drop_duplicates(subset=["item_id"], keep="first")
    
    item_to_sid = df_mapping.set_index("item_id")["sid"].to_dict()
    item_to_sid_parts = df_mapping.set_index("item_id")[sid_cols].apply(tuple, axis=1).to_dict()
    
    candidate_pool = df_mapping[["sid", "sid1", "sid2", "sid3"]].apply(tuple, axis=1).tolist()
    
    sid1_pool = df_mapping["sid1"].unique().tolist()
    sid2_pool = df_mapping["sid2"].unique().tolist()
    sid3_pool = df_mapping["sid3"].unique().tolist()
    
    existing_sids = set(df_mapping["sid"].tolist())
    
    logging.info(f"Successfully built mapping relationships with {len(item_to_sid)} valid item_ids")
    logging.info(f"sid candidate pool size: {len(candidate_pool)}")
    logging.info(f"sid1 pool size: {len(sid1_pool)} | sid2 pool size: {len(sid2_pool)} | sid3 pool size: {len(sid3_pool)}")
    logging.info(f"Total existing valid sids: {len(existing_sids)}")
    
    return item_to_sid, item_to_sid_parts, candidate_pool, sid1_pool, sid2_pool, sid3_pool, existing_sids

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
    item_to_sid: Dict[int, str],
    item_to_sid_parts: Dict[int, Tuple[str, str, str]],
    missing_id_handling: str = "skip"
) -> pd.DataFrame:
    """Convert int-type ids in userseq to sid and sid1/sid2/sid3."""
    logging.info("Starting id->sid mapping conversion...")
    
    def map_history_to_sid(history_list: List[int]) -> Optional[str]:
        sid_list = [item_to_sid.get(item_id) for item_id in history_list]
        valid_sid_list = [sid for sid in sid_list if sid is not None]
        
        if missing_id_handling == "error" and len(valid_sid_list) != len(history_list):
            raise ValueError(f"Unmapped ids found in history item_id: {[item_id for item_id, sid in zip(history_list, sid_list) if sid is None]}")
        
        return ",".join(valid_sid_list) if valid_sid_list else None
    
    df_userseq["history_sid_str"] = df_userseq["history_item_list"].apply(map_history_to_sid)
    df_userseq["chosen"] = df_userseq["item_id"].map(item_to_sid)
    df_userseq["sid_parts"] = df_userseq["item_id"].map(item_to_sid_parts)
    
    df_userseq[["sid1", "sid2", "sid3"]] = pd.DataFrame(
        df_userseq["sid_parts"].tolist(),
        index=df_userseq.index
    )
    
    before_filter = len(df_userseq)
    df_userseq = df_userseq.dropna(subset=["history_sid_str", "chosen", "sid1", "sid2", "sid3"])
    after_filter = len(df_userseq)
    
    logging.info(f"Mapping conversion complete: {before_filter} original rows, {after_filter} valid rows (filtered out rows without valid mappings)")
    
    return df_userseq[["history_sid_str", "chosen", "sid1", "sid2", "sid3"]]

def generate_rejected_sid(
    chosen_sid1: str,
    chosen_sid2: str,
    chosen_sid3: str,
    chosen_sid: str,
    candidate_pool: List[Tuple[str, str, str, str]],
    rule: int,
    sid1_pool: Optional[List[str]] = None,
    sid2_pool: Optional[List[str]] = None,
    sid3_pool: Optional[List[str]] = None,
    existing_sids: Optional[set[str]] = None,
    sid_separator: str = ""
) -> Optional[str]:
    """Generate a rejected sid based on the specified rule (0=invalid random, 1=sid1 differs, 2=sid2 differs, 3=sid3 differs)."""
    if rule == 0:
        if not all([sid1_pool, sid2_pool, sid3_pool, existing_sids]):
            logging.error("When using rule 0, sid1_pool, sid2_pool, sid3_pool and existing_sids must be provided")
            return None
        
        max_attempts = 100
        attempt = 0
        
        while attempt < max_attempts:
            rand_sid1 = random.choice(sid1_pool)
            rand_sid2 = random.choice(sid2_pool)
            rand_sid3 = random.choice(sid3_pool)
            
            new_sid = f"{rand_sid1}{sid_separator}{rand_sid2}{sid_separator}{rand_sid3}"
            
            if new_sid not in existing_sids and new_sid != chosen_sid:
                return new_sid
            
            attempt += 1
        
        logging.warning(f"After {max_attempts} attempts, could not generate a valid invalid sid (chosen: {chosen_sid})")
        return None
    
    valid_candidates = []
    vc_length = 0
    for cand_sid, cand_sid1, cand_sid2, cand_sid3 in candidate_pool:
        if cand_sid == chosen_sid:
            continue
        
        if rule == 1:
            if cand_sid1 != chosen_sid1:
                valid_candidates.append(cand_sid)
                vc_length += 1
        elif rule == 2:
            if cand_sid1 == chosen_sid1 and cand_sid2 != chosen_sid2:
                valid_candidates.append(cand_sid)
                vc_length += 1
        elif rule == 3:
            if cand_sid1 == chosen_sid1 and cand_sid2 == chosen_sid2 and cand_sid3 != chosen_sid3:
                valid_candidates.append(cand_sid)
                vc_length += 1
        if vc_length >= 50:
            break
    if valid_candidates:
        return random.choice(valid_candidates)
    else:
        return None

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
    candidate_pool: List[Tuple[str, str, str, str]],
    rejected_rule: int,
    sid1_pool: Optional[List[str]] = None,
    sid2_pool: Optional[List[str]] = None,
    sid3_pool: Optional[List[str]] = None,
    existing_sids: Optional[set[str]] = None,
    sid_separator: str = "_"
) -> None:
    """Convert data to DPO JSONL format with chosen/rejected fields."""
    logging.info(f"Starting to generate rejected field using rule {rejected_rule} and convert to JSONL format...")
    
    output_dir = os.path.dirname(jsonl_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    valid_samples = []
    for _, row in tqdm(df_converted.iterrows()):
        rejected_sid = generate_rejected_sid(
            chosen_sid1=row["sid1"],
            chosen_sid2=row["sid2"],
            chosen_sid3=row["sid3"],
            chosen_sid=row["chosen"],
            candidate_pool=candidate_pool,
            rule=rejected_rule,
            sid1_pool=sid1_pool,
            sid2_pool=sid2_pool,
            sid3_pool=sid3_pool,
            existing_sids=existing_sids,
            sid_separator=sid_separator
        )
        if rejected_sid:
            json_line = {
                "instruction": instruction,
                "input": row["history_sid_str"],
                "chosen": row["chosen"],
                "rejected": rejected_sid
            }
            valid_samples.append(json_line)
        else:
            logging.warning(f"No valid rejected sid for rule {rejected_rule}, skipping sample (chosen: {row['chosen']})")
    
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for sample in valid_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    logging.info(f"Successfully saved JSONL file to: {jsonl_path}")
    logging.info(f"JSONL file has {len(valid_samples)} valid records (filtered out samples without valid rejected)")

def main():
    parser = argparse.ArgumentParser(description="Process userseq data: id->sid mapping conversion and generate JSONL training data with chosen/rejected")
    parser.add_argument("--mapping_csv", default="/path/to/sid/checkpoint/itemid_to_sid.csv", help="Path to the mapping CSV containing item_id, sid, sid1, sid2, sid3 columns")
    parser.add_argument("--userseq_csv", default="/path/to/userseq/train/dataset.csv", help="Path to the input userseq.csv")
    parser.add_argument("--converted_csv_path", default="/path/to/userseq/train/dataset_temp.csv", help="Path to save intermediate converted data (CSV format)")
    parser.add_argument("--jsonl_path", default="/path/to/userseq/train/dataset_llm_dpo_stage0.jsonl", help="Path to the output JSONL file")
    parser.add_argument("--instruction", default="You are a professional search recommendation engine. Please predict the semantic id of the next item to be visited based on the user's historical visit records. The following are the historical records:", help="Instruction field content in JSONL")
    parser.add_argument(
        "--missing_id_handling",
        choices=["skip", "error"],
        default="skip",
        help="How to handle unmapped item_ids: skip=skip the row, error=raise exception (default: skip)"
    )
    parser.add_argument(
        "--rejected_rule",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="Rejected field generation rule: 0=invalid sid (random sid1/sid2/sid3 combination, not existing sid); 1=sid1 differs; 2=sid1 same, sid2 differs; 3=sid1/sid2 same, sid3 differs"
    )
    parser.add_argument(
        "--sid_separator",
        type=str,
        default="",
        help="sid concatenation separator (only used for rule 0, must match existing sid format, default: empty)"
    )
    
    args = parser.parse_args()
    
    for path in [args.mapping_csv, args.userseq_csv]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")
    
    try:
        logging.info("="*60)
        logging.info("Starting to load mapping dictionary and candidate pool...")
        
        item_to_sid, item_to_sid_parts, candidate_pool, sid1_pool, sid2_pool, sid3_pool, existing_sids = load_mapping_dict(args.mapping_csv)
        
        logging.info("\n" + "="*60)
        logging.info("Starting to load and parse userseq.csv...")
        df_userseq = load_userseq_csv(args.userseq_csv)
        
        logging.info("\n" + "="*60)
        df_converted = convert_userseq_data(
            df_userseq=df_userseq,
            item_to_sid=item_to_sid,
            item_to_sid_parts=item_to_sid_parts,
            missing_id_handling=args.missing_id_handling
        )
        
        logging.info("\n" + "="*60)
        save_converted_csv(df_converted, args.converted_csv_path)
        
        logging.info("\n" + "="*60)
        convert_to_jsonl(
            df_converted=df_converted,
            jsonl_path=args.jsonl_path,
            instruction=args.instruction,
            candidate_pool=candidate_pool,
            rejected_rule=args.rejected_rule,
            sid1_pool=sid1_pool,
            sid2_pool=sid2_pool,
            sid3_pool=sid3_pool,
            existing_sids=existing_sids,
            sid_separator=args.sid_separator
        )
        
        logging.info("\n" + "="*60)
        logging.info("All processes completed!")
        with open(args.jsonl_path, "r", encoding="utf-8") as f:
            jsonl_count = len(f.readlines())
        logging.info("Final output:")
        logging.info(f"  - Intermediate converted data: {args.converted_csv_path} ({len(df_converted)} rows)")
        logging.info(f"  - JSONL training data: {args.jsonl_path} ({jsonl_count} valid samples)")
    except Exception as e:
        logging.error(f"\nPipeline execution failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()