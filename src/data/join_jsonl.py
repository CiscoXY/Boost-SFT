import json
import random
import os
from typing import List, Dict, Set, Tuple


def load_jsonl_file(file_path: str) -> Tuple[List[Dict], int]:
    data = []
    line_count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json_obj = json.loads(line)
                    data.append(json_obj)
                    line_count += 1
                except json.JSONDecodeError as e:
                    print(f"Warning: file {file_path} line {line_num} has invalid JSON format, skipped: {e}")
        return data, line_count
    except Exception as e:
        print(f"Error: failed to read file {file_path}: {e}")
        raise


def check_field_consistency(all_data: List[Tuple[List[Dict], str]]) -> Set[str]:
    if not all_data:
        raise ValueError("No valid data files to process")

    base_fields = None
    for data, file_path in all_data:
        if data:
            base_fields = set(data[0].keys())
            print(f"Using file {file_path} fields as baseline: {sorted(base_fields)}")
            break

    if not base_fields:
        raise ValueError("All input files have no valid data")

    for data, file_path in all_data:
        for idx, sample in enumerate(data):
            sample_fields = set(sample.keys())
            if sample_fields != base_fields:
                missing = base_fields - sample_fields
                extra = sample_fields - base_fields
                error_msg = f"Field inconsistency: file {file_path} line {idx+1}"
                if missing:
                    error_msg += f" missing fields: {sorted(missing)}"
                if extra:
                    error_msg += f" extra fields: {sorted(extra)}"
                raise ValueError(error_msg)

    print("All files pass field consistency check")
    return base_fields


def merge_jsonl_files(
    input_paths: List[str], 
    output_path: str, 
    seed: int = 42,
    k: int = 0
) -> None:
    """Merge multiple JSONL files, optionally sampling k samples from each."""
    if k < 0:
        print(f"Error: k value cannot be negative (current k={k})")
        return
    
    all_data = []
    file_stats = []
    total_samples = 0
    random.seed(seed)

    print("Starting to read input files and sample...")
    for path in input_paths:
        if not os.path.exists(path):
            print(f"Warning: file {path} does not exist, skipped")
            continue
        if not os.path.isfile(path):
            print(f"Warning: {path} is not a file, skipped")
            continue
        
        data, original_count = load_jsonl_file(path)
        file_name = os.path.basename(path)
        
        if k == 0:
            sampled_data = data
            sampled_count = original_count
        else:
            if k >= original_count:
                sampled_data = data
                sampled_count = original_count
                print(f"   Note: {file_name} original sample count ({original_count}) <= k ({k}), using all samples")
            else:
                sampled_data = random.sample(data, k=k)
                sampled_count = k
        
        all_data.append((sampled_data, path))
        file_stats.append((file_name, original_count, sampled_count))
        total_samples += sampled_count
        print(f"   Processed {file_name}: original {original_count} -> sampled {sampled_count}")

    if not all_data:
        print("Error: no valid readable JSONL files found")
        return

    try:
        base_fields = check_field_consistency(all_data)
    except ValueError as e:
        print(f"Field check failed: {e}")
        return

    merged_data = []
    for data, _ in all_data:
        merged_data.extend(data)

    print(f"Starting shuffle of {total_samples} merged samples...")
    random.shuffle(merged_data)
    print("Shuffle complete")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    print(f"Starting to write output file {output_path}...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in merged_data:
                sorted_sample = dict(sorted(sample.items()))
                f.write(json.dumps(sorted_sample, ensure_ascii=False) + '\n')
        print(f"Output file saved to: {output_path}")
    except Exception as e:
        print(f"Failed to write output file: {e}")
        return

    print("\nMerge Statistics Report")
    print("-" * 60)
    print(f"Total input files: {len(file_stats)}")
    print(f"Sampling parameter k: {k} (0 means extract all)")
    print("Per-file processing details:")
    for file_name, original_count, sampled_count in file_stats:
        print(f"  - {file_name}: original {original_count} -> sampled {sampled_count}")
    print(f"Total merged samples: {total_samples}")
    print(f"Field set: {sorted(base_fields)}")
    print(f"Field count: {len(base_fields)}")
    print(f"Shuffle seed: {seed}")
    print(f"Output file path: {output_path}")
    print("-" * 60)


if __name__ == "__main__":
    INPUT_FILES = [
        "/path/to/userseq/train/dataset_llm.jsonl",
        "/path/to/sid_understand/sid_understand_dataset.jsonl",
    ]
    OUTPUT_FILE = "/path/to/llm_data/mix/dataset/train_mix_200_samples.jsonl"
    SHUFFLE_SEED = 42
    K = 200

    print("Starting JSONL file merge task...")
    merge_jsonl_files(
        input_paths=INPUT_FILES,
        output_path=OUTPUT_FILE,
        seed=SHUFFLE_SEED,
        k=K
    )
    print("\nTask completed!")