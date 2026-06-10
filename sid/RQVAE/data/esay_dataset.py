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
        # 预留embedding的完整内存，这样加载数据快
        embedding_size=(140_0000, 2560),
        record_path = None,
        data_dir = None,
        pt_files = None,
    ):
        """
        Args:
            dim_cutoff : embedding的截断
            embedding_size : 预定义的embedding的完整内存
            record_path : 记录日志的路径
            Note：
                1. 如果data_dir和pt_files都为None，则无法加载数据
                2. 如果data_dir不为None，则从data_dir中加载.pt文件
                3. 如果pt_files不为None，则从pt_files中加载.pt文件
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
        write_record_log(f"预定义 {max_num_items} × {embedding_dim} embedding tensor 耗时: {(time.time() - process_start_time):.3f} s" , log_path=self.record_path)

        start_time = time.time()
        if self.data_dir is not None:
            pt_files = glob.glob(str(Path(self.data_dir) / "*.pt"))
        elif self.pt_files is not None:
            pt_files = self.pt_files
        else:
            write_record_log(f"未指定数据目录或.pt文件列表，无法加载数据" , log_path=self.record_path)
            raise ValueError("未指定数据目录或.pt文件列表，无法加载数据")
        pt_files.sort()
        
        if not pt_files:
            raise ValueError(f"在 {self.data_dir} 目录下未找到任何.pt文件")
        write_record_log(f"找到 {len(pt_files)} 个.pt文件，排序耗时: {(time.time() - start_time):.3f} s" , log_path=self.record_path)

        current_position = 0
        for i, pt_file in enumerate(pt_files):
            load_start = time.time()
            try:
                # 包含"parent_asin"和"embedding"的字典
                pt_data = torch.load(pt_file)
                
                if "item_id" not in pt_data or "embedding" not in pt_data:
                    raise ValueError(f"文件 {pt_file} 必须包含'parent_asin'和'embedding'键")
                
                parent_asin_chunk = pt_data["item_id"]
                if isinstance(parent_asin_chunk, torch.Tensor):
                    sku_chunk = parent_asin_chunk.numpy().astype(str)
                else:  # 若为list直接转numpy
                    sku_chunk = np.array(parent_asin_chunk, dtype=str)

                emb_chunk = pt_data["embedding"]
                
                # if sku_chunk.dtype != torch.int64:
                #     raise ValueError(f"文件 {pt_file} 中的'sku'必须是int64类型，实际: {sku_chunk.dtype}")
                
                if emb_chunk.dim() != 2:
                    raise ValueError(f"文件 {pt_file} 中的'embedding'不是2维的，实际维度: {emb_chunk.dim()}")
                
                chunk_num, chunk_dim = emb_chunk.shape
                if chunk_dim != embedding_dim:
                    raise ValueError(
                        f"文件 {pt_file} 中的embedding维度不匹配，"
                        f"预期: {embedding_dim}, 实际: {chunk_dim}"
                    )
                
                if len(sku_chunk) != chunk_num:
                    raise ValueError(
                        f"文件 {pt_file} 中的'sku'和'embedding'数量不匹配，"
                        f"sku数量: {len(sku_chunk)}, embedding数量: {chunk_num}"
                    )
                
                # 检查是否有足够空间容纳当前chunk
                if current_position + chunk_num > max_num_items:
                    adjust_num = max_num_items - current_position
                    sku_chunk = sku_chunk[:adjust_num]
                    emb_chunk = emb_chunk[:adjust_num]
                    chunk_num = adjust_num
                    write_record_log(
                        f"警告: 预定义空间不足，调整当前chunk大小为 {chunk_num}, "
                        f"文件: {pt_file}" , log_path=self.record_path
                    )
                
                item_emb[current_position : current_position + chunk_num, :] = emb_chunk
                all_skus[current_position : current_position + chunk_num] = sku_chunk
                
                # 更新当前位置
                current_position += chunk_num
                
                write_record_log(
                    f"已加载 {i+1}/{len(pt_files)}: {pt_file}, "
                    f"大小: {chunk_num} × {chunk_dim}, "
                    f"当前位置: {current_position}, "
                    f"耗时: {(time.time() - load_start):.3f} s" , log_path=self.record_path
                )
                
                if current_position >= max_num_items:
                    write_record_log(f"已达到预定义的最大embedding数量 {max_num_items}，停止加载剩余文件" , log_path=self.record_path)
                    break
                
            except Exception as e:
                write_record_log(f"加载文件 {pt_file} 时出错: {str(e)}" , log_path=self.record_path)
                raise

        if current_position < max_num_items:
            write_record_log(
                f"INFO: 所有.pt文件加载完成，共填充了 {current_position} 个embedding，"
                f"小于预定义的 {max_num_items} 个，仅保留已填充部分" , log_path=self.record_path
            )
            # 截断到实际填充的大小
            item_emb = item_emb[:current_position, :]
            all_skus = all_skus[:current_position]
        else:
            write_record_log(
                f"INFO: 所有.pt文件加载完成，正好填充了预定义的 {max_num_items} 个embedding" , log_path=self.record_path
            )

        # 存储数据到HeteroData
        data["item"].x = item_emb
        data["item"].itemId_list = all_skus # 从pt文件中获取的sku作为itemId
        
        print(f"最终item_emb shape: {item_emb.shape}")
        print(f"最终itemId_list shape: {data['item'].itemId_list.shape}")
        print("itemId_list的dtype:", data["item"].itemId_list.dtype)
        write_record_log(f"最终item_emb shape: {item_emb.shape}" , log_path=self.record_path)
        write_record_log(f"最终itemId_list shape: {data['item'].itemId_list.shape}" , log_path=self.record_path)
        write_record_log(f"itemId_list的dtype: {data['item'].itemId_list.dtype}" , log_path=self.record_path)

        start_time = time.time()
        gen = torch.Generator()
        gen.manual_seed(42)
        data["item"].is_train = torch.rand(current_position, generator=gen) > 0.01  # 99%训练，1%评估
        write_record_log(f"生成训练/测试分割标记 耗时: {(time.time() - start_time):.3f} s" , log_path=self.record_path)
        print("=====train/eval split done=====")

        write_record_log(f"数据加载完成，总耗时: {(time.time() - process_start_time):.3f} s" , log_path=self.record_path)
        self.data = data

class TigerDataset_Direct(PreprocessingMixin):
    def __init__(
        self,
        embeddings,
        skus,
        dim_cutoff = 2560,
    ):
        # embedddings.shape = [len(skus) , embedding_dim]
        # skus是 nd.array(int64)
        self.data = HeteroData()
        self.data["item"].x = embeddings
        self.data["item"].itemId_list = skus
        gen = torch.Generator()
        gen.manual_seed(42)
        self.data["item"].is_train = torch.rand(len(skus), generator=gen) > 0.02  # 99%训练，1%评估

        self.dim_cutoff = dim_cutoff


class ItemData(Dataset):
    def __init__(self, raw_dataset, train_test_split):
        self.raw_data = raw_dataset.data["item"]["x"]  # 原始embedding（共享内存）
        self.raw_sku = raw_dataset.data["item"]["itemId_list"]  # 原始sku（共享内存）
        self.dim_cutoff = raw_dataset.dim_cutoff
        
        # 整数索引替代布尔mask
        if train_test_split == "train":
            filt = raw_dataset.data["item"]["is_train"]
        elif train_test_split == "eval":
            filt = ~raw_dataset.data["item"]["is_train"]
        elif train_test_split == "all":
            filt = torch.ones(len(self.raw_data), dtype=bool)
        else:
            raise ValueError(f"Invalid train_test_split: {train_test_split}")
        
        # 仅存储索引，不复制数据
        self.indices = torch.nonzero(filt, as_tuple=False).squeeze(dim=1)  # 形状为 (N,) 的int tensor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 通过索引动态获取数据（不复制，仅引用）
        raw_idx = self.indices[idx]
        x = self.raw_data[raw_idx, : self.dim_cutoff]
        sku_id = self.raw_sku[raw_idx]
        item_ids = raw_idx.detach().clone().unsqueeze(0) if not isinstance(idx, torch.Tensor) else raw_idx.unsqueeze(0)
        return ItemBatch(ids=item_ids, x=x, sku_id=sku_id)


def create_infinite_dataloader(dataloader):
    """创建无限循环的dataloader迭代器"""
    while True:
        for batch in dataloader:
            yield batch

class TemplatedTextDataset(Dataset):
    def __init__(self, 
                file_path: str, 
                text_template: str,
                custom_fields: List[str]):
        """
        初始化数据集
        :param file_path: 输入文件路径
        :param text_template: 文本模板，使用{字段名}作为占位符
        :param custom_fields: 需要保留的自定义字段列表
        """
        self.file_path = file_path
        self.text_template = text_template
        self.custom_fields = custom_fields
        # 存储数据、模板应用错误数、总行数
        self.data, self.template_error_count, self.total_lines = self._load_and_process_data()  

    def _load_and_process_data(self) -> tuple[List[Dict], int, int]:
        """修改返回值：(数据列表, 模板应用错误行数, 总行数)"""
        file_ext = os.path.splitext(self.file_path)[-1].lower()
        
        if file_ext == ".json":
            return self._load_json()
        elif file_ext == ".parquet":
            return self._load_parquet()
        else:
            raise ValueError(f"不支持的文件格式：{file_ext}，仅支持 .json 和 .parquet")

    def _apply_template(self, item: Dict) -> tuple[str, bool]:
        """
        应用模板并返回结果和是否成功
        :return: (格式化文本, 成功标志) 失败时返回("", False)
        """
        try:
            formatted_text = self.text_template.format_map(
                {k: str(v).strip() if v is not None else "" for k, v in item.items()}
            )
            return formatted_text, True
        except KeyError as e:
            print(f"警告：模板中包含不存在的字段 {e}，当前条目实际值：{item}")
            return "", False  # 失败时返回空文本和False

    def _extract_template_variables(self) -> List[str]:
        """从模板中提取所有占位符字段名"""
        return re.findall(r'\{(.*?)\}', self.text_template)

    def _load_json(self) -> tuple[List[Dict], int, int]:
        """加载JSON文件并应用template，返回(数据, 模板错误行数, 总行数)"""
        data = []
        template_error_count = 0  # 模板应用错误计数
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
                    print(f"警告：第{line_num}行JSON解析错误，内容：{line[:100]}... 错误原因：{e}")
                    continue 
                
                # 应用模板并检查是否成功
                formatted_text, is_success = self._apply_template(item)
                if not is_success:
                    template_error_count += 1 
                    continue 
                
                # 收集数据（仅保留成功且非空的文本）
                if formatted_text.strip():
                    metadata = {"text": formatted_text}
                    for field in self.custom_fields:
                        metadata[field] = item.get(field, None)
                    data.append(metadata)
        
        print(f"load JSON done，有效数据：{len(data)}/{total_lines}，模板应用错误行数：{template_error_count}/{total_lines}")
        return data, template_error_count, total_lines

    def _load_parquet(self) -> tuple[List[Dict], int, int]:
        """加载Parquet文件并应用template，返回(数据, 模板错误行数, 总行数)"""
        data = []
        template_error_count = 0  # 模板应用错误计数
        df = pd.read_parquet(self.file_path)
        total_rows = len(df)  # 总行数
        
        for row_num, row in enumerate(tqdm(df.iterrows(), total=total_rows, desc=f"load Parquet: {self.file_path}"), 1):
            _, row_data = row
            item = row_data.to_dict()
            
            # 应用模板并检查是否成功
            formatted_text, is_success = self._apply_template(item)
            if not is_success:
                template_error_count += 1 
                continue 
            
            # 收集数据（仅保留成功且非空的文本）
            if formatted_text.strip():
                metadata = {"text": formatted_text}
                for field in self.custom_fields:
                    value = item.get(field)
                    metadata[field] = value if pd.notna(value) else None
                data.append(metadata)
        
        print(f"load Parquet done，有效数据：{len(data)}/{total_rows}，模板应用错误行数：{template_error_count}/{total_rows}")
        return data, template_error_count, total_rows

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]
    
    def get_metadata(self) -> pd.DataFrame:
        """获取所有元数据（含自定义字段）"""
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
        """获取数据集信息"""
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

        # 如果是tensor，转换为list以便后续处理
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
