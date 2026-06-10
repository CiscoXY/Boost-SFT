from data.schemas import ItemBatch, SeqBatch
import torch
from datetime import datetime
from typing import Optional
import os

def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data

def item_batch_to(batch, device):
    processed_fields = []
    for _, value in batch._asdict().items():
        # 仅对Tensor类型to(device)（ids和x），字符串sku_id保留
        if isinstance(value, torch.Tensor):
            processed_fields.append(value.to(device))
        else:
            processed_fields.append(value)
    return ItemBatch(*processed_fields)

def seq_batch_to(batch, device):
    return SeqBatch(*[v.to(device) for _, v in batch._asdict().items()])


def next_item_batch(dataloader, device):
    batch = next(dataloader)
    return item_batch_to(batch, device)


def next_seq_batch(dataloader, device):
    batch = next(dataloader)
    return seq_batch_to(batch, device)

def write_record_log(record_str: str , log_path: Optional[str] = None):
    """写入记录数到日志文件"""
    if log_path is None:
        log_path = "/mnt/rqvae日志.log"
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_content = f"{current_time} : {record_str}"
    
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(log_content + '\n')