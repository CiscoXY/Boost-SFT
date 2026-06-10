import numpy as np
import pandas as pd
import os
import random
import polars as pl
import glob
import torch
from pathlib import Path
from data.preprocessing import PreprocessingMixin
from torch_geometric.data import HeteroData
from data.schemas import ItemBatch, SeqBatch
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict
from tiger_utils.data_utils import load_from_path
from data.utils import write_record_log
import time
import re
from tqdm import tqdm
import json

class TigerDataset(PreprocessingMixin):
    def __init__(
        self,
        dim_cutoff,
        # Pre-allocate full embedding memory for faster data loading
        embedding_size=(140_0000, 2560),
        record_path = None,
        data_dir = None,
        pt_files = None,
    ):
        """
        Args:
            dim_cutoff : embedding truncation dimension
            embedding_size : pre-defined full embedding memory size
            record_path : log file path
            Note:
                1. If both data_dir and pt_files are None, no data can be loaded
                2. If data_dir is not None, load .pt files from data_dir
                3. If pt_files is not None, load .pt files from pt_files list
        """
        self.data_dir = data_dir
        self.pt_files = pt_files
        self.dim_cutoff = dim_cutoff
        self.embedding_size = embedding_size
        self.record_path = record_path
        self.process()

    def process(self) -> None:
        data = HeteroData()

        process_start_time = time.time()
        max_num_items, embedding_dim = self.embedding_size
        item_emb = torch.zeros(max_num_items, embedding_dim, dtype=torch.float32)
        all_skus = np.zeros(max_num_items, dtype=object)
        write_record_log(f"Pre-allocated {max_num_items} x {embedding_dim} embedding tensor, time: {(time.time() - process_start_time):.3f} s" , log_path=self.record_path)

        start_time = time.time()
        if self.data_dir is not None:
            pt_files = glob.glob(str(Path(self.data_dir) / "*.pt"))
        elif self.pt_files is not None:
            pt_files = self.pt_files
        else:
            write_record_log(f"No data directory or .pt file list specified, cannot load data" , log_path=self.record_path)
            raise ValueError("No data directory or .pt file list specified, cannot load data")
        pt_files.sort()
        
        if not pt_files:
            raise ValueError(f"No .pt files found in directory: {self.data_dir}")
        write_record_log(f"Found {len(pt_files)} .pt files, sort time: {(time.time() - start_time):.3f} s" , log_path=self.record_path)

        current_position = 0
        for i, pt_file in enumerate(pt_files):
            load_start = time.time()
            try:
                # Dictionary containing "parent_asin" and "embedding"
                pt_data = torch.load(pt_file)
                
                if "item_id" not in pt_data or "embedding" not in pt_data:
                    raise ValueError(f"File {pt_file} must contain 'parent_asin' and 'embedding' keys")
                
                parent_asin_chunk = pt_data["item_id"]
                if isinstance(parent_asin_chunk, torch.Tensor):
                    sku_chunk = parent_asin_chunk.numpy().astype(str)
                else:  # if list, convert to numpy directly
                    sku_chunk = np.array(parent_asin_chunk, dtype=str)

                emb_chunk = pt_data["embedding"]
                
                # if sku_chunk.dtype != torch.int64:
                #     raise ValueError(f"File {pt_file}: 'sku' must be int64 type, actual: {sku_chunk.dtype}")
                
                if emb_chunk.dim() != 2:
                    raise ValueError(f"File {pt_file}: 'embedding' is not 2-dimensional, actual dims: {emb_chunk.dim()}")
                
                chunk_num, chunk_dim = emb_chunk.shape
                if chunk_dim != embedding_dim:
                    raise ValueError(
                        f"File {pt_file}: embedding dimension mismatch, "
                        f"expected: {embedding_dim}, actual: {chunk_dim}"
                    )
                
                if len(sku_chunk) != chunk_num:
                    raise ValueError(
                        f"File {pt_file}: 'sku' and 'embedding' count mismatch, "
                        f"sku count: {len(sku_chunk)}, embedding count: {chunk_num}"
                    )
                
                # Check if there is enough space for the current chunk
                if current_position + chunk_num > max_num_items:
                    adjust_num = max_num_items - current_position
                    sku_chunk = sku_chunk[:adjust_num]
                    emb_chunk = emb_chunk[:adjust_num]
                    chunk_num = adjust_num
                    write_record_log(
                        f"Warning: pre-allocated space insufficient, adjusted chunk size to {chunk_num}, "
                        f"file: {pt_file}" , log_path=self.record_path
                    )
                
                item_emb[current_position : current_position + chunk_num, :] = emb_chunk
                all_skus[current_position : current_position + chunk_num] = sku_chunk
                
                # Update current position
                current_position += chunk_num
                
                write_record_log(
                    f"Loaded {i+1}/{len(pt_files)}: {pt_file}, "
                    f"size: {chunk_num} x {chunk_dim}, "
                    f"current position: {current_position}, "
                    f"time: {(time.time() - load_start):.3f} s" , log_path=self.record_path
                )
                
                if current_position >= max_num_items:
                    write_record_log(f"Reached pre-defined max embedding count {max_num_items}, stopping loading remaining files" , log_path=self.record_path)
                    break
                
            except Exception as e:
                write_record_log(f"Error loading file {pt_file}: {str(e)}" , log_path=self.record_path)
                raise

        if current_position < max_num_items:
            write_record_log(
                f"INFO: All .pt files loaded, filled {current_position} embeddings, "
                f"less than pre-defined {max_num_items}, keeping only filled portion" , log_path=self.record_path
            )
            # Truncate to actual filled size
            item_emb = item_emb[:current_position, :]
            all_skus = all_skus[:current_position]
        else:
            write_record_log(
                f"INFO: All .pt files loaded, exactly filled pre-defined {max_num_items} embeddings" , log_path=self.record_path
            )

        # Store data to HeteroData
        data["item"].x = item_emb
        data["item"].itemId_list = all_skus  # SKU from pt files used as itemId
        
        print(f"Final item_emb shape: {item_emb.shape}")
        print(f"Final itemId_list shape: {data['item'].itemId_list.shape}")
        print("itemId_list dtype:", data["item"].itemId_list.dtype)
        write_record_log(f"Final item_emb shape: {item_emb.shape}" , log_path=self.record_path)
        write_record_log(f"Final itemId_list shape: {data['item'].itemId_list.shape}" , log_path=self.record_path)
        write_record_log(f"itemId_list dtype: {data['item'].itemId_list.dtype}" , log_path=self.record_path)

        start_time = time.time()
        gen = torch.Generator()
        gen.manual_seed(42)
        data["item"].is_train = torch.rand(current_position, generator=gen) > 0.01  # 99% train, 1% eval
        write_record_log(f"Generating train/test split mask, time: {(time.time() - start_time):.3f} s" , log_path=self.record_path)
        print("=====train/eval split done=====")

        write_record_log(f"Data loading complete, total time: {(time.time() - process_start_time):.3f} s" , log_path=self.record_path)
        self.data = data

class TigerDataset_Direct(PreprocessingMixin):
    def __init__(
        self,
        embeddings,
        skus,
        dim_cutoff = 2560,
    ):
        # embedddings.shape = [len(skus), embedding_dim]
        # skus is nd.array(int64)
        self.data = HeteroData()
        self.data["item"].x = embeddings
        self.data["item"].itemId_list = skus
        gen = torch.Generator()
        gen.manual_seed(42)
        self.data["item"].is_train = torch.rand(len(skus), generator=gen) > 0.02  # 99% train, 1% eval

        self.dim_cutoff = dim_cutoff


class ItemData(Dataset):
    def __init__(self, raw_dataset, train_test_split):
        self.raw_data = raw_dataset.data["item"]["x"]  # raw embeddings (shared memory)
        self.raw_sku = raw_dataset.data["item"]["itemId_list"]  # raw SKUs (shared memory)
        self.dim_cutoff = raw_dataset.dim_cutoff
        
        # Integer index instead of boolean mask
        if train_test_split == "train":
            filt = raw_dataset.data["item"]["is_train"]
        elif train_test_split == "eval":
            filt = ~raw_dataset.data["item"]["is_train"]
        elif train_test_split == "all":
            filt = torch.ones(len(self.raw_data), dtype=bool)
        else:
            raise ValueError(f"Invalid train_test_split: {train_test_split}")
        
        # Store only indices, don't copy data
        self.indices = torch.nonzero(filt, as_tuple=False).squeeze(dim=1)  # shape (N,) int tensor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Dynamically get data via index (no copy, just reference)
        raw_idx = self.indices[idx]
        x = self.raw_data[raw_idx, : self.dim_cutoff]
        sku_id = self.raw_sku[raw_idx]
        item_ids = raw_idx.detach().clone().unsqueeze(0) if not isinstance(idx, torch.Tensor) else raw_idx.unsqueeze(0)
        return ItemBatch(ids=item_ids, x=x, sku_id=sku_id)


def create_infinite_dataloader(dataloader):
    """Create an infinite loop dataloader iterator"""
    while True:
        for batch in dataloader:
            yield batch

class TemplatedTextDataset(Dataset):
    def __init__(self, 
                file_path: str, 
                text_template: str,
                custom_fields: List[str]):
        """
        Initialize dataset
        :param file_path: input file path
        :param text_template: text template, using {field_name} as placeholders
        :param custom_fields: list of custom fields to preserve
        """
        self.file_path = file_path
        self.text_template = text_template
        self.custom_fields = custom_fields
        # Store data, template error count, total lines
        self.data, self.template_error_count, self.total_lines = self._load_and_process_data()  

    def _load_and_process_data(self) -> tuple[List[Dict], int, int]:
        """Return value: (data list, template error lines, total lines)"""
        file_ext = os.path.splitext(self.file_path)[-1].lower()
        
        if file_ext == ".json":
            return self._load_json()
        elif file_ext == ".parquet":
            return self._load_parquet()
        else:
            raise ValueError(f"Unsupported file format: {file_ext}, only .json and .parquet are supported")

    def _apply_template(self, item: Dict) -> tuple[str, bool]:
        """
        Apply template and return result and success flag
        :return: (formatted text, success flag) returns ("", False) on failure
        """
        try:
            formatted_text = self.text_template.format_map(
                {k: str(v).strip() if v is not None else "" for k, v in item.items()}
            )
            return formatted_text, True
        except KeyError as e:
            print(f"Warning: template contains non-existent field {e}, current item values: {item}")
            return "", False  # Return empty text and False on failure

    def _extract_template_variables(self) -> List[str]:
        """Extract all placeholder field names from template"""
        return re.findall(r'\{(.*?)\}', self.text_template)

    def _load_json(self) -> tuple[List[Dict], int, int]:
        """Load JSON file and apply template, return (data, template error lines, total lines)"""
        data = []
        template_error_count = 0  # template application error count
        with open(self.file_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
        
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(tqdm(f, total=total_lines, desc=f"loading JSON: {self.file_path}"), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)  
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON parse error at line {line_num}, content: {line[:100]}... error: {e}")
                    continue 
                
                # Apply template and check if successful
                formatted_text, is_success = self._apply_template(item)
                if not is_success:
                    template_error_count += 1 
                    continue 
                
                # Collect data (only keep successful and non-empty text)
                if formatted_text.strip():
                    metadata = {"text": formatted_text}
                    for field in self.custom_fields:
                        metadata[field] = item.get(field, None)
                    data.append(metadata)
        
        print(f"load JSON done, valid data: {len(data)}/{total_lines}, template error lines: {template_error_count}/{total_lines}")
        return data, template_error_count, total_lines

    def _load_parquet(self) -> tuple[List[Dict], int, int]:
        """Load Parquet file and apply template, return (data, template error lines, total lines)"""
        data = []
        template_error_count = 0  # template application error count
        df = pd.read_parquet(self.file_path)
        total_rows = len(df)  # total rows
        
        for row_num, row in enumerate(tqdm(df.iterrows(), total=total_rows, desc=f"load Parquet: {self.file_path}"), 1):
            _, row_data = row
            item = row_data.to_dict()
            
            # Apply template and check if successful
            formatted_text, is_success = self._apply_template(item)
            if not is_success:
                template_error_count += 1 
                continue 
            
            # Collect data (only keep successful and non-empty text)
            if formatted_text.strip():
                metadata = {"text": formatted_text}
                for field in self.custom_fields:
                    value = item.get(field)
                    metadata[field] = value if pd.notna(value) else None
                data.append(metadata)
        
        print(f"load Parquet done, valid data: {len(data)}/{total_rows}, template error lines: {template_error_count}/{total_rows}")
        return data, template_error_count, total_rows

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]
    
    def get_metadata(self) -> pd.DataFrame:
        """Get all metadata (including custom fields)"""
        return pd.DataFrame(self.data)

class RawDataset:
    def __init__(self, data_path: str = None, data_list: List[Dict] = None):
        if data_list is not None:
            self.dataset = HFDataset.from_list(data_list)
        elif data_path is not None:
            self.dataset = load_from_path(data_path)
        else:
            raise ValueError("Either data_path or data_list must be provided")

        self.dataset = self.dataset.with_format("torch")

    def __getitem__(self, key: str):
        if key == "userId":
            return self.dataset["userId"]
        elif key == "itemId":
            return self.dataset["itemId"]
        elif key == "itemId_fut":
            return self.dataset["label"]
        else:
            raise KeyError(f"Key {key} not found")

    def __len__(self):
        return len(self.dataset)

    def get_file_info(self) -> Dict:
        """Get dataset info"""
        return {
            "total_records": len(self.dataset),
            "columns": self.dataset.column_names,
            "features": self.dataset.features,
        }


class SeqData(Dataset):
    def __init__(
        self,
        raw_dataset: RawDataset,
        max_seq_len: int,
        is_train: bool,
        subsample: bool,
    ) -> None:
        assert (not subsample) or is_train, "Can only subsample on training split."
        self.subsample = subsample
        self.max_seq_len = max_seq_len
        self.is_train = is_train
        self._preprocess_data(raw_dataset)

    def _preprocess_data(self, raw_dataset: RawDataset):
        self.user_ids = raw_dataset["userId"]
        self.item_ids = raw_dataset["itemId"]
        self.item_ids_fut = raw_dataset["itemId_fut"]

        # If tensor, convert to list for subsequent processing
        if hasattr(self.user_ids, "tolist"):
            self.user_ids = self.user_ids.tolist()
        if hasattr(self.item_ids, "tolist"):
            self.item_ids = self.item_ids.tolist()
        if hasattr(self.item_ids_fut, "tolist"):
            self.item_ids_fut = self.item_ids_fut.tolist()

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx: int):
        user_id = self.user_ids[idx]
        item_seq = self.item_ids[idx]
        item_fut = self.item_ids_fut[idx]

        if not isinstance(item_seq, list):
            item_seq = item_seq.tolist() if hasattr(item_seq, "tolist") else [item_seq]
        if not isinstance(item_fut, list):
            item_fut = item_fut.tolist() if hasattr(item_fut, "tolist") else [item_fut]

        if self.subsample and len(item_seq) > 0:
            full_seq = item_seq + item_fut
            if len(full_seq) >= 3:
                start_idx = random.randint(0, max(0, len(full_seq) - 3))
                max_end = min(len(full_seq), start_idx + self.max_seq_len + 1)
                end_idx = random.randint(start_idx + 2, max_end)
                sample = full_seq[start_idx:end_idx]

                item_ids = sample[:-1]
                item_ids_fut = [sample[-1]]
            else:
                item_ids = item_seq
                item_ids_fut = item_fut
        else:
            item_ids = item_seq
            item_ids_fut = item_fut

        if len(item_ids) > self.max_seq_len:
            item_ids = item_ids[-self.max_seq_len :]

        item_ids_tensor = torch.tensor(item_ids, dtype=torch.long)
        item_ids_fut_tensor = torch.tensor(item_ids_fut, dtype=torch.long)
        user_id_tensor = torch.tensor(user_id, dtype=torch.long)

        return {
            "user_ids": user_id_tensor,
            "ids": item_ids_tensor,
            "ids_fut": item_ids_fut_tensor,
            "seq_mask": (item_ids_tensor >= 0),
        }


def fast_collate_fn(batch):
    user_ids = torch.stack([item["user_ids"] for item in batch])
    ids_fut = torch.stack([item["ids_fut"] for item in batch])

    sequences = [item["ids"] for item in batch]
    padded_ids = pad_sequence(sequences, batch_first=True, padding_value=-1)

    seq_masks = []
    max_len = padded_ids.size(1)

    for item in batch:
        seq_len = len(item["ids"])
        mask = torch.cat(
            [
                torch.ones(seq_len, dtype=torch.bool),
                torch.zeros(max_len - seq_len, dtype=torch.bool),
            ]
        )
        seq_masks.append(mask)

    seq_masks = torch.stack(seq_masks)

    return SeqBatch(
        user_ids=user_ids.unsqueeze(1),
        ids=padded_ids,
        ids_fut=ids_fut,
        seq_mask=seq_masks,
    )


def create_fast_dataloader(
    dataset: SeqData,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=fast_collate_fn,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else 2,
    )
