import json
import jsonlines
from collections import defaultdict
from typing import List
from datasets import Dataset as HFDataset, load_dataset, concatenate_datasets
from data.utils import write_record_log
from typing import List
import glob
from pathlib import Path
import gc
import time

def process_rq_vae_codebook(semantic_ids_list, sku_list):
    item_to_tokens = {}
    tokens_to_item = defaultdict(list)
    for idx, (sku_id, semantic_ids) in enumerate(zip(sku_list, semantic_ids_list)):
        # Build SKU-to-tokens mapping
        tokens = []
        for layer_idx, semantic_id in zip('abcdefg', semantic_ids):
            token = f"<{layer_idx}_{semantic_id}>"
            tokens.append(token)
        item_to_tokens[sku_id] = tokens
        
        # Build tokens-to-SKU mapping
        tokens_key = tuple(tokens)
        tokens_to_item[tokens_key].append(sku_id)
    
    tokens_to_item = dict(tokens_to_item)
    
    return (
        item_to_tokens,
        tokens_to_item,
    )


def process_for_train_data(train_data, item2tokens):
    def items_to_token_string(items):
        """Convert item list to token string"""
        token_string = ""
        for item in items:
            if item in item2tokens:
                tokens = item2tokens[item]
                token_string += "".join(map(str, tokens)) + ","
            else:
                print(f"Warning: Item {item} not found in mapping")
        return token_string.rstrip(",")

    def labels_to_token_string(labels):
        """Convert label list to token string"""
        token_string = ""
        for label in labels:
            if label in item2tokens:
                tokens = item2tokens[label]
                token_string += "".join(map(str, tokens))
            else:
                print(f"Warning: Label {label} not found in mapping")
        return token_string

    # Generate training data
    prompts = []
    for item in train_data:
        user_id = item["userId"]
        item_ids = item["itemId"]
        labels = item["label"]
        train_token_string = items_to_token_string(item_ids)
        label_token_string = labels_to_token_string(labels)
        user_content = f"User ID:{user_id}, this user has clicked the following items in chronological order: {train_token_string}, can you predict the next item the user might click?"
        conversation = {
            "conversations": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": label_token_string},
            ]
        }
        prompts.append(conversation)

    return prompts


def load_multiple_files(file_paths, file_type="json") -> HFDataset:
    combined_dataset = None  # Initialize merged dataset as None

    for i , file_path in enumerate(file_paths):
        print(f"Loading: {file_path}")
        try:
            if file_type == "json":
                dataset = load_dataset("json", data_files=file_path, split="train")
            elif file_type == "parquet":
                dataset = load_dataset("parquet", data_files=file_path, split="train")
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            print(f"  -> Loaded {len(dataset)} records")
            
            # Concatenate after loading each file
            if combined_dataset is None:
                combined_dataset = dataset
            else:
                start_time =  time.time()
                combined_dataset = concatenate_datasets([combined_dataset, dataset])
                print(f" Merge time: {(time.time() - start_time):.3f}s")
                print(f"  -> Combined dataset size after this file: {len(combined_dataset)} records")
                write_record_log(f"File {i+1}: {file_path} merge time: {(time.time() - start_time):.2f} s")
            del dataset 
            gc.collect()

        except Exception as e:
            print(f"  -> Error loading {file_path}: {e}")
            continue

    if combined_dataset is None:
        raise ValueError("No files were successfully loaded")

    print(f"Final combined dataset size: {len(combined_dataset)} records")
    return combined_dataset


def process_for_alignment_data(train_data, item2tokens):
    training_prompts_text2token = []
    training_prompts_token2text = []

    for item in train_data:
        item_id = item["itemId"]
        item_description = item["text"]

        if item_id not in item2tokens:
            continue

        special_tokens = item2tokens[item_id]

        tokens_str = "".join(special_tokens)

        prompt_1 = {
            "instruction": f"What is the description of item {tokens_str}?",
            "output": item_description,
        }
        training_prompts_token2text.append(prompt_1)

        prompt_2 = {
            "instruction": f"Which item has the description: {item_description}",
            "output": tokens_str,
        }
        training_prompts_text2token.append(prompt_2)

    return training_prompts_text2token, training_prompts_token2text


def save_training_data(prompts, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)


def find_files_by_extensions(directory: Path, extensions: List[str]) -> List[str]:
    files = []
    for ext in extensions:
        pattern = str(directory / f"*{ext}")
        files.extend(glob.glob(pattern))
    return sorted(files)


def load_from_directory(directory: Path) -> HFDataset:
    json_files = find_files_by_extensions(directory, [".json", ".jsonl"])
    if json_files:
        print(f"Found {len(json_files)} JSON files, processing...")
        return load_multiple_files(json_files, file_type="json")
    parquet_files = find_files_by_extensions(directory, [".parquet"])
    if parquet_files:
        print(f"Found {len(parquet_files)} Parquet files, processing...")
        write_record_log(f"Found {len(parquet_files)} Parquet files, processing...")
        return load_multiple_files(parquet_files, file_type="parquet")
    raise ValueError(
        f"No supported files (.json, .jsonl, .parquet) found in directory: {directory}"
    )


def load_single_file(data_file: str) -> HFDataset:
    if data_file.endswith(".parquet"):
        return load_dataset("parquet", data_files=data_file, split="train")
    elif data_file.endswith((".json", ".jsonl")):
        return load_dataset("json", data_files=data_file, split="train")
    else:
        raise ValueError(f"Unsupported file format: {data_file}")


def load_from_path(data_path: str) -> HFDataset:
    path = Path(data_path)

    if path.is_file():
        return load_single_file(str(path))
    elif path.is_dir():
        return load_from_directory(path)
    else:
        raise ValueError(f"Path does not exist: {data_path}")


def load_and_prepare_test_data(test_data, item2tokens):
    def items_to_token_string(items):
        token_string = ""
        for item in items:
            if item in item2tokens:
                tokens = item2tokens[item]
                token_string += "".join(map(str, tokens)) + ","
            else:
                print(f"Warning: Item {item} not found in mapping")
        return token_string.rstrip(",")

    test_prompts = []
    for item in test_data:
        user_id = item["userId"]
        item_ids = item["itemId"]
        test_token_string = items_to_token_string(item_ids)
        prompt_content = f"User ID:{user_id}, this user has clicked the following items in chronological order: {test_token_string}, can you predict the next item the user might click?"

        test_item = {"prompt": prompt_content}
        test_prompts.append(test_item)

    return test_prompts
