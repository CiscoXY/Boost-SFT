import os
import sys
import random
import json
import logging
import math
from typing import List, Optional, Dict, Union, Tuple
from dataclasses import dataclass, field

import fire
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset, concatenate_datasets, Dataset
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from transformers.trainer import Trainer
from torch.utils.tensorboard import SummaryWriter

# 固定全局种子
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)


@dataclass
class TrainingConfig:
    """配置类，向后兼容，新增衰减参数"""
    train_on_inputs: bool = False  # 最终状态：是否 mask input
    output_token_weights: Optional[List[float]] = None  # 最终状态的 output weights
    
    # 新增：Non-Rec 数据采样衰减参数
    k1: int = 0  # 前 k1 个 epoch Non-Rec 保持 100%
    m1: int = 0  # 到 m1 个 epoch Non-Rec 衰减到 0%，0表示不衰减
    
    # 新增：Rec 任务 weight 衰减参数
    k2: int = 0  # 前 k2 个 epoch 标准 SFT
    m2: int = 0  # 到 m2 个 epoch 达到最终状态，0表示直接最终状态


class DynamicMixedSampler(Sampler):
    """
    支持动态调整 Non-Rec 样本采样权重的 Sampler
    """
    def __init__(self, dataset, indices, is_rec_list, k1, m1, seed=42):
        self.dataset = dataset
        self.indices = indices
        self.is_rec_list = is_rec_list
        self.k1 = k1
        self.m1 = m1
        self.seed = seed
        self.epoch = 0
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        
        # 分离 Rec 和 Non-Rec 索引
        self.rec_indices = [idx for idx, is_rec in zip(indices, is_rec_list) if is_rec]
        self.non_rec_indices = [idx for idx, is_rec in zip(indices, is_rec_list) if not is_rec]
        
        self.total_samples = len(self.rec_indices) + len(self.non_rec_indices)
        self.rec_count = len(self.rec_indices)
        self.non_rec_count = len(self.non_rec_indices)
        
    def set_epoch(self, epoch):
        self.epoch = epoch
        self.generator.manual_seed(self.seed + epoch)
        
        # 计算并打印当前采样率
        ratio = self._get_current_ratio()
        actual_non_rec = int(self.non_rec_count * ratio)
        total_used = self.rec_count + actual_non_rec
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} - Non-Rec Data Sampling")
        print(f"{'='*60}")
        print(f"  Rec 样本数:     {self.rec_count} (始终 100%)")
        print(f"  Non-Rec 样本数: {self.non_rec_count} (原始)")
        print(f"  Non-Rec 采样率: {ratio*100:.1f}%")
        print(f"  Non-Rec 实际使用: {actual_non_rec}")
        print(f"  总训练样本数:   {total_used} / {self.total_samples} ({total_used/self.total_samples*100:.1f}%)")
        
        if epoch < self.k1:
            print(f"  [阶段: 保持期 (epoch < {self.k1})]")
        elif epoch >= self.m1:
            print(f"  [阶段: 衰减完成 (epoch >= {self.m1})]")
        else:
            progress = (epoch - self.k1) / (self.m1 - self.k1)
            print(f"  [阶段: 衰减中 ({self.k1} -> {self.m1}), 进度: {progress*100:.1f}%]")
        print(f"{'='*60}\n")
        sys.stdout.flush()
        
    def _get_current_ratio(self):
        """计算当前 Non-Rec 采样率"""
        if self.m1 <= self.k1 or self.epoch < self.k1:
            return 1.0
        elif self.epoch >= self.m1:
            return 0.0
        else:
            return 1.0 - (self.epoch - self.k1) / (self.m1 - self.k1)
        
    def __iter__(self):
        ratio = self._get_current_ratio()
        
        candidates = self.rec_indices.copy()
        
        if ratio > 0 and len(self.non_rec_indices) > 0:
            n_sample = int(len(self.non_rec_indices) * ratio)
            if n_sample > 0:
                perm = torch.randperm(len(self.non_rec_indices), generator=self.generator).tolist()
                selected = [self.non_rec_indices[i] for i in perm[:n_sample]]
                candidates.extend(selected)
        
        candidates_tensor = torch.tensor(candidates)
        shuffled = candidates_tensor[torch.randperm(len(candidates), generator=self.generator)].tolist()
        
        return iter(shuffled)
    
    def __len__(self):
        # 返回当前 epoch 的实际长度
        ratio = self._get_current_ratio()
        return len(self.rec_indices) + int(len(self.non_rec_indices) * ratio)


class WeightedCELossTrainer(Trainer):
    """
    自定义 Trainer：
    1. 支持 Rec 任务 Loss Weight 随 epoch 动态衰减
    2. Eval 时固定使用最终状态计算 Loss，并计算 Hit Rate
    """
    
    def __init__(self, *args, training_config: TrainingConfig = None, 
                eval_samples_to_print: Optional[List[Dict]] = None, 
                train_sampler: Optional[Sampler] = None,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.training_config = training_config  # 确保这行存在
        self.eval_samples_to_print = eval_samples_to_print or []
        self.current_epoch = 0
        self.train_sampler = train_sampler
        
    def get_train_dataloader(self):
        """重写以使用自定义 DynamicMixedSampler"""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        if self.train_sampler is not None:
            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                sampler=self.train_sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=0,  # 必须设为 0
                pin_memory=self.args.dataloader_pin_memory,
            )
        
        return super().get_train_dataloader()
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        动态 Loss 计算：
        - Non-Rec：始终标准 SFT（所有 weight=1）
        - Rec：根据 current_epoch 动态调整 weight
          * < k2: 标准 SFT
          * k2 -> m2: 线性过渡
          * >= m2: 使用最终配置（train_on_inputs, output_token_weights）
        """
        labels = inputs.pop("labels")
        input_lens = inputs.pop("input_lens", None)  # [batch]
        is_rec_flags = inputs.pop("is_rec", None)    # [batch]
        
        # 前向传播
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        batch_size, seq_len = shift_labels.shape
        
        # 初始化 weights
        shift_weights = torch.ones_like(shift_labels, dtype=torch.float32)
        
        if input_lens is not None and is_rec_flags is not None:
            for b in range(batch_size):
                if not is_rec_flags[b].item():  # Non-Rec 任务
                    continue  # weight 保持 1.0
                
                # Rec 任务：计算当前 epoch 的 weight
                input_len = max(0, input_lens[b].item() - 1)  # -1 because of shift
                
                # 计算进度
                k2, m2 = self.training_config.k2, self.training_config.m2
                if m2 <= k2 or self.current_epoch < k2:
                    # 标准 SFT 阶段
                    w_input, w_output_list = 1.0, None
                elif self.current_epoch >= m2:
                    # 最终状态
                    w_input = 0.0 if not self.training_config.train_on_inputs else 1.0
                    w_output_list = self.training_config.output_token_weights
                else:
                    # 线性插值阶段
                    progress = (self.current_epoch - k2) / (m2 - k2)
                    
                    # Input weight: 1.0 -> final
                    final_input_w = 0.0 if not self.training_config.train_on_inputs else 1.0
                    w_input = 1.0 + progress * (final_input_w - 1.0)
                    
                    # Output weights: 1.0 -> target
                    w_output_list = self.training_config.output_token_weights
                
                # 应用 input weight
                if input_len > 0:
                    shift_weights[b, :input_len] = w_input
                
                # 应用 output weight
                if input_len < seq_len:
                    output_len = seq_len - input_len
                    if w_output_list is None:
                        # 标准 SFT：所有 output token weight = 1
                        pass  # 已经是 1.0
                    else:
                        # 应用特定 weights
                        for i in range(output_len):
                            if i < len(w_output_list):
                                target_w = w_output_list[i]
                            else:
                                target_w = 0.0  # 超出部分为 0
                            
                            # 如果是插值阶段，需要计算当前值
                            if k2 <= self.current_epoch < m2:
                                progress = (self.current_epoch - k2) / (m2 - k2)
                                current_w = 1.0 + progress * (target_w - 1.0)
                            else:
                                current_w = target_w
                            
                            pos = input_len + i
                            if pos < seq_len:
                                shift_weights[b, pos] = current_w
        
        # Flatten 并计算 loss
        vocab_size = shift_logits.size(-1)
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        flat_weights = shift_weights.view(-1)
        
        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        losses = loss_fct(flat_logits, flat_labels)
        weighted_losses = losses * flat_weights
        
        total_weight = flat_weights.sum()
        if total_weight > 0:
            loss = weighted_losses.sum() / total_weight
        else:
            loss = weighted_losses.sum()
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """
        Eval 逻辑：
        1. Loss 计算：Rec 固定使用最终状态（input=0/masked, output=target weights）
        2. 计算 Hit Rate @1/2/3（仅针对 Rec 样本）
        """
        # 保存当前 epoch，强制设为 m2 以触发最终状态
        original_epoch = self.current_epoch
        self.current_epoch = max(self.current_epoch, self.training_config.m2)
        
        # 标准评估（计算 loss）
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # 计算 Hit Rate（恢复原始 epoch，因为 Hit Rate 计算不依赖 weight）
        self.current_epoch = original_epoch
        hit_rates = self._compute_hit_rates(eval_dataset)
        output.update(hit_rates)
        
        print(f"\n{'='*70}")
        print(f"Evaluation Hit Rate Results (Epoch {self.current_epoch:.2f})")
        print(f"{'='*70}")
        for key, value in hit_rates.items():
            if 'hit_rate' in key:
                print(f"  {key}: {value*100:.2f}%")
            else:
                print(f"  {key}: {value}")
        print(f"{'='*70}\n")
        sys.stdout.flush()
        # 打印样本
        self._print_eval_samples()
        
        # 恢复 epoch
        self.current_epoch = original_epoch
        
        return output
    
    def _compute_hit_rates(self, eval_dataset) -> Dict[str, float]:
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
                
        self.model.eval()
        device = self.args.device
        dataloader = self.get_eval_dataloader(eval_dataset)
        
        original_padding_side = self.processing_class.padding_side
        self.processing_class.padding_side = 'left'
        
        if self.processing_class.pad_token_id is None:
            self.processing_class.pad_token_id = self.processing_class.eos_token_id
        
        correct = [0, 0, 0]
        total_rec = 0
        
        try:
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                is_rec = batch.get("is_rec", torch.zeros(input_ids.size(0)))
                
                rec_mask = is_rec == 1
                if not rec_mask.any():
                    continue
                    
                rec_input_ids = input_ids[rec_mask]
                rec_attention_mask = attention_mask[rec_mask]
                rec_labels = labels[rec_mask]
                
                batch_size = rec_input_ids.size(0)
                
                # 逐个样本处理（因为每个样本的 target_start 可能不同）
                for i in range(batch_size):
                    total_rec += 1
                    
                    # 找到该样本 output 的开始位置（第一个非 -100 的 label）
                    valid_positions = (rec_labels[i] != -100).nonzero(as_tuple=True)[0]
                    
                    if len(valid_positions) == 0:
                        continue
                    
                    target_start = valid_positions[0].item()
                    
                    # 关键修复：只取 input 部分（0 到 target_start）传给 generate
                    input_part = rec_input_ids[i][:target_start].unsqueeze(0)
                    mask_part = rec_attention_mask[i][:target_start].unsqueeze(0)
                    
                    # 确保至少有一些 input
                    if mask_part.sum() == 0:
                        continue
                    
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            input_ids=input_part,
                            attention_mask=mask_part,
                            max_new_tokens=3,
                            do_sample=False,
                            pad_token_id=self.processing_class.pad_token_id,
                            eos_token_id=self.processing_class.eos_token_id,
                        )
                    
                    # generated_ids 长度 = input_part 长度 + 3
                    # 取生成的 3 个 token
                    generated = generated_ids[0][target_start:target_start+3]
                    target = rec_labels[i][target_start:target_start+3]
                    
                    # 过滤 -100（以防万一）
                    valid_mask = target != -100
                    target = target[valid_mask]
                    generated = generated[:len(target)]
                    
                    if len(target) == 0:
                        continue
                    
                    # 统计 Hit Rate
                    if len(target) >= 1 and generated[0] == target[0]:
                        correct[0] += 1
                    if len(target) >= 2 and torch.equal(generated[:2], target[:2]):
                        correct[1] += 1
                    if len(target) >= 3 and torch.equal(generated[:3], target[:3]):
                        correct[2] += 1
        
        finally:
            self.processing_class.padding_side = original_padding_side
        
        result = {}
        for i, k in enumerate([1, 2, 3]):
            key = f"eval_rec_hit_rate@{k}"
            result[key] = correct[i] / total_rec if total_rec > 0 else 0.0
        
        result["eval_rec_total_samples"] = total_rec
        return result
    
    def _print_eval_samples(self):
        if not self.eval_samples_to_print:
            return
            
        print(f"\n{'='*80}")
        print(f"Step {self.state.global_step} - Sample Predictions")
        print(f"{'='*80}")
        
        self.model.eval()
        device = self.args.device
        
        # 修复：临时切换为 left-padding
        original_padding_side = self.processing_class.padding_side
        self.processing_class.padding_side = 'left'
        
        try:
            for idx, sample in enumerate(self.eval_samples_to_print):
                gen_prompt = sample["generation_prompt"]
                
                inputs = self.processing_class(
                    gen_prompt, 
                    return_tensors="pt", 
                    truncation=True,
                    max_length=self.args.max_seq_length
                ).to(device)
                
                with torch.no_grad():
                    generate_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=16,
                        do_sample=False,
                        pad_token_id=self.processing_class.pad_token_id,
                        eos_token_id=self.processing_class.eos_token_id,
                    )
                
                input_length = inputs["input_ids"].shape[1]
                generated_tokens = generate_ids[0][input_length:]
                pred_text = self.processing_class.decode(
                    generated_tokens, 
                    skip_special_tokens=True
                )
                
                print(f"\n[Sample {idx+1}/{len(self.eval_samples_to_print)}]")
                print(f"Input:    {sample['raw_input'][:300]}{'...' if len(sample['raw_input']) > 300 else ''}")
                print(f"Output:   {sample['raw_output'][:300]}{'...' if len(sample['raw_output']) > 300 else ''}")
                print(f"Predict:  {pred_text[:300]}{'...' if len(pred_text) > 300 else ''}")
                print("-" * 80)
        finally:
            # 恢复 padding_side
            self.processing_class.padding_side = original_padding_side
        
        print(f"{'='*80}\n")


class EpochUpdateCallback(TrainerCallback):
    """在每个 epoch 开始时更新配置并打印参数"""
    
    def __init__(self):
        self.trainer = None  # 将在外部设置
        self.last_printed_epoch = -1
        
    def set_trainer(self, trainer):
        """设置 trainer 引用"""
        self.trainer = trainer
        
    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """每个 epoch 开始时调用"""
        if self.trainer is None:
            return control
            
        current_epoch = int(state.epoch)
        
        # 避免重复打印同一个 epoch
        if current_epoch == self.last_printed_epoch:
            return control
            
        self.last_printed_epoch = current_epoch
        self.trainer.current_epoch = current_epoch
        
        # 1. 更新 Sampler 并打印 Non-Rec 采样信息
        if hasattr(self.trainer, "train_sampler") and self.trainer.train_sampler is not None:
            self.trainer.train_sampler.set_epoch(current_epoch)
        
        # 2. 打印 Rec Weight 配置
        config = self.trainer.training_config
        k2, m2 = config.k2, config.m2
        
        print(f"\n{'='*70}")
        print(f"Epoch {current_epoch} - Rec Task Loss Weight Configuration")
        print(f"{'='*70}")
        
        if m2 <= k2 or current_epoch < k2:
            # 标准 SFT 阶段
            w_input = 1.0
            w_output_desc = "[1.0, 1.0, 1.0...] (标准 SFT)"
            stage = f"标准 SFT期 (epoch < {k2})"
        elif current_epoch >= m2:
            # 最终状态
            w_input = 0.0 if not config.train_on_inputs else 1.0
            w_output_desc = f"{config.output_token_weights} (最终配置)"
            stage = f"最终状态期 (epoch >= {m2})"
        else:
            # 线性插值阶段
            progress = (current_epoch - k2) / (m2 - k2)
            final_input = 0.0 if not config.train_on_inputs else 1.0
            w_input = 1.0 + progress * (final_input - 1.0)  
            
            if config.output_token_weights:
                current_weights = [round(1.0 + progress * (t - 1.0), 2) for t in config.output_token_weights]
                w_output_desc = f"{current_weights} (插值中，目标: {config.output_token_weights})"
            else:
                w_output_desc = "[1.0, 1.0...]"
            stage = f"衰减过渡期 ({k2} -> {m2}, 进度: {progress*100:.1f}%)"
        
        if abs(w_input - 0.0) < 0.001:
            status = "0.000 (mask)"
        elif abs(w_input - 1.0) < 0.001:
            status = "1.000 (full)"
        else:
            status = f"{w_input:.3f} (decaying)"
        print(f"  当前阶段:        {stage}")
        print(f"  Input Weight:    {status}")
        print(f"  Output Weights:  {w_output_desc}")
        print(f"{'='*70}\n")
        sys.stdout.flush()
        
        return control


def is_recommendation_task(instruction: str) -> bool:
    """判断是否为推荐任务"""
    rec_keywords = [
        "search recommendation", 
        "predict the semantic id",
        "recommendation engine",
        "next item"
    ]
    return any(kw in instruction.lower() for kw in rec_keywords)


def generate_and_tokenize_prompt(data_point: Dict, tokenizer, cutoff_len: int):
    """
    生成 prompt 并 tokenize
    策略：input 和 output 各至少保留 100 tokens，总和不超过 cutoff_len
    超长时截断 input 的后面（保留前面）和 output 的后面（保留前面）
    """
    instruction = data_point["instruction"]
    input_text = data_point.get("input", "")
    output = data_point.get("output", "")
    
    is_rec = is_recommendation_task(instruction)
    
    # Tokenize 各部分
    input_only_prompt = generate_qwen_prompt({**data_point, "output": ""})
    input_only_ids = tokenizer(
        input_only_prompt,
        truncation=False,
        return_tensors=None,
    )["input_ids"]
    
    output_ids = tokenizer(
        output,
        add_special_tokens=False,
        return_tensors=None,
    )["input_ids"] if output else []
    
    eos_token_id = tokenizer.eos_token_id
    
    input_len = len(input_only_ids)
    output_len = len(output_ids)
    total_needed = input_len + output_len + 1  # +1 for EOS
    
    MIN_LEN = 100
    
    if total_needed <= cutoff_len:
        # 不超长：正常组合
        input_ids = input_only_ids + output_ids
        if len(input_ids) < cutoff_len:
            input_ids.append(eos_token_id)
        actual_input_len = input_len
    else:
        # 超长：需要截断，确保各至少保留 MIN_LEN
        available = cutoff_len - 1  # 留给 input + output 的空间（预留 EOS）
        
        if input_len <= MIN_LEN and output_len <= MIN_LEN:
            # 两者都短但总和超了（不应该发生）
            input_keep = input_len
            output_keep = output_len
        elif input_len <= MIN_LEN:
            # input 短，output 长：input 全保留，output 截断
            input_keep = input_len
            output_keep = min(output_len, available - input_keep)
            output_keep = max(MIN_LEN, output_keep)
            if input_keep + output_keep > available:
                output_keep = available - input_keep
        elif output_len <= MIN_LEN:
            # output 短，input 长：output 全保留，input 截断
            output_keep = output_len
            input_keep = min(input_len, available - output_keep)
            input_keep = max(MIN_LEN, input_keep)
            if input_keep + output_keep > available:
                input_keep = available - output_keep
        else:
            # 两者都长：各保留至少 MIN_LEN，剩余空间给 input
            output_keep = min(output_len, max(MIN_LEN, available - MIN_LEN))
            input_keep = available - output_keep
            
            # 如果 input 不够 MIN_LEN，从 output 借空间
            if input_keep < MIN_LEN:
                input_keep = MIN_LEN
                output_keep = available - MIN_LEN
                if output_keep < MIN_LEN:
                    print(f"WARNING: cutoff_len {cutoff_len} too small for MIN_LEN {MIN_LEN}")
                    output_keep = MIN_LEN
        
        # 执行截断策略：
        # Input：从后面截断，保留前面（input_only_ids[:input_keep]）
        # Output：从后面截断，保留前面（output_ids[:output_keep]）
        truncated_input = input_only_ids[:input_keep] if input_keep < input_len else input_only_ids
        truncated_output = output_ids[:output_keep] if output_keep < output_len else output_ids
        
        input_ids = truncated_input + truncated_output
        if len(input_ids) < cutoff_len:
            input_ids.append(eos_token_id)
        
        actual_input_len = len(truncated_input)
        
        if input_keep < input_len or output_keep < output_len:
            print(f"INFO: Truncated input {input_len}->{input_keep} (keep front), "
                  f"output {output_len}->{output_keep} (keep front), "
                  f"total={len(input_ids)} (cutoff={cutoff_len})")
    
    # 构建 attention_mask 和 labels
    attention_mask = [1] * len(input_ids)
    labels = input_ids.copy()
    
    # Mask input 部分
    for i in range(min(actual_input_len, len(labels))):
        labels[i] = -100
    
    # 安全检查
    valid_count = sum(1 for l in labels if l != -100)
    if valid_count == 0:
        print(f"WARNING: All labels are -100! input_len={actual_input_len}, total_len={len(labels)}")
        if len(labels) > 0:
            labels[-1] = input_ids[-1]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "input_lens": actual_input_len,
        "is_rec": 1 if is_rec else 0,
    }


def prepare_eval_samples(raw_dataset: Dataset, tokenizer, num_samples: int = 5) -> List[Dict]:
    """准备 eval 样本"""
    indices = list(range(len(raw_dataset)))
    rng = random.Random(42)
    rng.shuffle(indices)
    selected_indices = indices[:num_samples]
    
    samples = []
    for idx in selected_indices:
        item = raw_dataset[idx]
        raw_input = item.get("input", "")
        raw_output = item.get("output", "")
        input_prompt_for_generation = generate_qwen_prompt({**item, "output": ""})
        
        samples.append({
            "raw_input": raw_input,
            "raw_output": raw_output,
            "generation_prompt": input_prompt_for_generation,
        })
    
    return samples


def train(
    # model/data params（保持原有字段）
    base_model: str = "Qwen/Qwen3-0.6B",
    train_data_path: List[str] = None,
    val_data_path: List[str] = None,
    output_dir: str = "./qwen3-training",
    sample: int = -1,
    seed: int = 42,
    # training hyperparams（保持原有字段）
    batch_size: int = 128,
    micro_batch_size: int = 12,
    num_epochs: int = 10,
    learning_rate: float = 1e-5,
    cutoff_len: int = 512,
    # custom loss params（保持原有字段，作为最终状态）
    train_on_inputs: bool = False,
    output_token_weights: Optional[List[float]] = None,
    # 新增：Non-Rec 数据衰减参数
    k1: int = 0,  # 前 k1 个 epoch Non-Rec 保持 100%
    m1: int = 0,  # 到 m1 个 epoch Non-Rec 衰减到 0，默认 0 表示不衰减
    # 新增：Rec 任务 weight 衰减参数
    k2: int = 0,  # 前 k2 个 epoch 标准 SFT（所有 weight=1）
    m2: int = 0,  # 到 m2 个 epoch 达到最终状态，默认 0 表示直接使用最终状态
    # llm hyperparams（保持原有字段）
    resume_from_checkpoint: str = None,
    gradient_checkpointing: bool = True,
    # eval params（保持原有字段）
    eval_samples_print_num: int = 5,
):
    """
    主训练函数
    新增支持：
    1. Non-Rec 数据采样率随 epoch 线性衰减（k1, m1）
    2. Rec 任务 Loss Weight 随 epoch 线性衰减（k2, m2）
    3. Eval 时固定使用最终状态 Loss 并计算 Hit Rate @1/2/3
    """
    assert train_data_path is not None and len(train_data_path) > 0, "必须提供训练数据路径"
    if val_data_path is None:
        val_data_path = []
    
    if resume_from_checkpoint in ["", "None", "none", "XXX"]:
        resume_from_checkpoint = None
    if isinstance(train_on_inputs, str):
        train_on_inputs = train_on_inputs.lower() in ('true', '1', 'yes', 'on')
        print(f"转换后的 train_on_inputs: {train_on_inputs} (类型: {type(train_on_inputs)})")
    
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    # 设备与精度
    device_map = "auto"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # 配置（向后兼容）
    config = TrainingConfig(
        train_on_inputs=train_on_inputs,
        output_token_weights=output_token_weights,
        k1=k1,
        m1=m1,
        k2=k2,
        m2=m2,
    )
    
    print(f"配置：train_on_inputs={train_on_inputs}, output_token_weights={output_token_weights}")
    if k1 > 0 or m1 > 0:
        print(f"Non-Rec 数据衰减：前 {k1} epoch 100%，到第 {m1} epoch 衰减至 0%")
    if k2 > 0 or m2 > 0:
        print(f"Rec Weight 衰减：前 {k2} epoch 标准 SFT，到第 {m2} epoch 达到最终状态")
    
    # ==================== 模型加载（保持原有逻辑）====================
    print("=== 加载模型 ===")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device_map,
        use_safetensors=True,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        pad_token='<|endoftext|>'
    )
    tokenizer.padding_side = "right"
    tokenizer.save_pretrained(output_dir)
    
    if gradient_checkpointing:
        print("启用 Gradient Checkpointing...")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # ==================== 数据加载与处理 ====================
    print("=== 加载数据集 ===")
    all_train_ds = [load_dataset("json", data_files=p)["train"] for p in train_data_path]
    raw_train_ds = concatenate_datasets(all_train_ds)
    if sample > -1:
        raw_train_ds = raw_train_ds.select(range(min(sample, len(raw_train_ds))))
    
    # Tokenize（保留 is_rec 和 input_lens）
    print("Tokenizing...")
    train_data = raw_train_ds.map(
        lambda x: generate_and_tokenize_prompt(x, tokenizer, cutoff_len),
        remove_columns=raw_train_ds.column_names,
        desc="Tokenizing train",
    )
    
    # 收集 is_rec 列表用于 sampler
    is_rec_list = train_data["is_rec"]
    
    # 验证集
    raw_val_ds = None
    val_data = None
    eval_samples = []
    if val_data_path:
        all_val_ds = [load_dataset("json", data_files=p)["train"] for p in val_data_path]
        raw_val_ds = concatenate_datasets(all_val_ds)
        val_data = raw_val_ds.map(
            lambda x: generate_and_tokenize_prompt(x, tokenizer, cutoff_len),
            remove_columns=raw_val_ds.column_names,
            desc="Tokenizing val",
        )
        eval_samples = prepare_eval_samples(raw_val_ds, tokenizer, eval_samples_print_num)
    
    # 设置格式
    columns_to_keep = ["input_ids", "attention_mask", "labels", "input_lens", "is_rec"]
    train_data = train_data.remove_columns([c for c in train_data.column_names if c not in columns_to_keep])
    train_data.set_format(type="torch")
    
    if val_data:
        val_data = val_data.remove_columns([c for c in val_data.column_names if c not in columns_to_keep])
        val_data.set_format(type="torch")
    
    # ==================== 自定义 Sampler ====================
    # 如果设置了衰减参数，使用 DynamicMixedSampler，否则使用默认
    train_sampler = None
    if k1 > 0 or m1 > 0:
        train_sampler = DynamicMixedSampler(
            train_data, 
            list(range(len(train_data))), 
            is_rec_list, 
            k1, m1, 
            seed
        )
        print(f"使用 DynamicMixedSampler，Non-Rec 将在 epoch {k1} 到 {m1} 间线性衰减")
    
    # ==================== Data Collator ====================
    class WeightedDataCollator(DataCollatorForSeq2Seq):
        def __call__(self, features):
            # 处理 list/tensor 转换
            for feature in features:
                for key in ["input_ids", "attention_mask", "labels", "input_lens", "is_rec"]:
                    if key in feature:
                        val = feature[key]
                        if isinstance(val, torch.Tensor):
                            feature[key] = val.cpu().item() if val.dim() == 0 else val.cpu().tolist()
                        elif isinstance(val, np.ndarray):
                            feature[key] = val.tolist()
            
            # 保留特殊字段
            input_lens = [f.pop("input_lens", 0) for f in features]
            is_rec = [f.pop("is_rec", 0) for f in features]
            
            # 调用父类
            batch = super().__call__(features)
            
            # 恢复字段
            batch["input_lens"] = torch.tensor(input_lens, dtype=torch.long)
            batch["is_rec"] = torch.tensor(is_rec, dtype=torch.long)
            
            return batch
    
    collator = WeightedDataCollator(
        tokenizer,
        padding=True,
        return_tensors="pt",
        pad_to_multiple_of=8
    )
    
    # ==================== 训练参数 ====================
    use_bf16 = torch.cuda.is_bf16_supported()
    
    training_args = TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_strategy="steps",
        logging_steps=5,
        eval_strategy="steps" if val_data else "no",
        eval_steps=400,
        save_strategy="steps",
        save_steps=400,
        output_dir=output_dir,
        save_total_limit=7,
        load_best_model_at_end=True if val_data else False,
        metric_for_best_model="eval_loss" if val_data else None,
        greater_is_better=False,
        report_to="tensorboard",
        logging_dir=f"{output_dir}/logs",
        remove_unused_columns=False,  # 重要：保留 input_lens 和 is_rec
        weight_decay=0.0,
        max_grad_norm=1.0,
        seed=seed,
        dataloader_num_workers=0 if train_sampler else 4,  # 自定义 sampler 需要 0
    )
    
    training_args.max_seq_length = cutoff_len
    
    epoch_callback = EpochUpdateCallback()
    trainer = WeightedCELossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
        data_collator=collator,
        training_config=config,
        eval_samples_to_print=eval_samples,
        train_sampler=train_sampler,
    )
    epoch_callback.set_trainer(trainer)
    trainer.add_callback(epoch_callback)
    
    # ===== 手动触发 Epoch 0 的打印 =====
    print("\n" + "="*70)
    print("训练开始前的初始配置检查")
    print("="*70)
    
    # 设置初始 epoch
    trainer.current_epoch = 0
    if train_sampler:
        train_sampler.set_epoch(0)  # 这会打印 Non-Rec 采样信息
    
    # 手动打印 Rec Weight 配置（复制 callback 的逻辑）
    k2, m2 = config.k2, config.m2
    epoch = 0
    
    print(f"\nEpoch {epoch} - Rec Task Loss Weight Configuration")
    print("-" * 70)
    
    if m2 <= k2 or epoch < k2:
        w_input = 1.0
        w_output_desc = "[1.0, 1.0, 1.0...] (标准 SFT)"
        stage = f"标准 SFT期 (epoch < {k2})"
    elif epoch >= m2:
        w_input = 0.0 if not config.train_on_inputs else 1.0
        target = config.output_token_weights
        w_output_desc = f"{target} (最终配置)"
        stage = f"最终状态期 (epoch >= {m2})"
    else:
        progress = (epoch - k2) / (m2 - k2)
        final_input = 0.0 if not config.train_on_inputs else 1.0
        w_input = 1.0 + progress * (final_input - 1.0)
        if config.output_token_weights:
            current_weights = [round(1.0 + progress * (t - 1.0), 2) for t in config.output_token_weights]
            w_output_desc = f"{current_weights} (插值中，目标: {config.output_token_weights})"
        else:
            w_output_desc = "[1.0, 1.0...] (无特殊配置)"
        stage = f"衰减过渡期 ({k2} -> {m2}, 进度: {progress*100:.1f}%)"
    
    print(f"  当前阶段:        {stage}")
    print(f"  Input Weight:    {w_input:.3f} ({'mask' if w_input==0 else 'full' if w_input==1 else 'partial'})")
    print(f"  Output Weights:  {w_output_desc}")
    print("="*70 + "\n")
    
    sys.stdout.flush()  # 强制刷新输出
    # ==========================================
    
    # 开始训练
    model.config.use_cache = False
    print("=== 正式开始训练 ===")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    print(f"保存模型到 {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("训练完成！")


def generate_qwen_prompt(data_point: Dict) -> str:
    """构建 Qwen3 ChatML 格式的 prompt（保持原有逻辑）"""
    instruction = data_point.get("instruction", "")
    input_text = data_point.get("input", "")
    output = data_point.get("output", "")
    
    system_content = "You are a helpful assistant."
    
    if input_text:
        user_content = f"{instruction}\n\nContext:\n{input_text}"
    else:
        user_content = instruction
    
    prompt = f"<|im_start|>system\n{system_content}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_content}<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n{output}"
    
    return prompt


if __name__ == "__main__":
    fire.Fire(train)