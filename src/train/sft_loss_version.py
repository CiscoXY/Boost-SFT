import os
import sys
import random
import json
import logging
from typing import List, Optional, Dict, Union
from dataclasses import dataclass

import fire
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset, concatenate_datasets, Dataset
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

# 固定全局种子以确保可复现性
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)


@dataclass
class TrainingConfig:
    """配置类，用于在 dataset map 中传递参数"""
    train_on_inputs: bool = False
    output_token_weights: Optional[List[float]] = None
    is_recommendation_task: bool = False  # 是否为 sid seq -> sid 推荐任务


class WeightedCELossTrainer(Trainer):
    """
    自定义 Trainer 支持以下功能：
    1. 对 input 部分 mask (label = -100)
    2. 对 output 部分应用自定义权重
    3. Eval 时打印 10 条样本的预测结果
    """
    
    def __init__(self, *args, eval_samples_to_print: Optional[List[Dict]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_samples_to_print = eval_samples_to_print or []
        self.global_step_tracker = 0
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        重写 loss 计算逻辑：
        - 使用 labels 区分 input/output
        - 使用 loss_weights 对 output token 加权
        - 对于 -100 的 label，同时考虑 weight=0
        """
        labels = inputs.pop("labels")
        loss_weights = inputs.pop("loss_weights", None)  # [batch, seq_len]
        
        # 前向传播
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Shift for next token prediction (预测下一个 token)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        if loss_weights is not None:
            shift_weights = loss_weights[..., 1:].contiguous()
        else:
            # 默认所有非 -100 位置权重为 1.0，-100 位置为 0
            shift_weights = (shift_labels != -100).float()
        
        # Flatten
        vocab_size = shift_logits.size(-1)
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        flat_weights = shift_weights.view(-1)
        
        # 计算每个 token 的 loss (reduction='none')
        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        losses = loss_fct(flat_logits, flat_labels)
        
        # 应用权重: 对于原本被忽略的位置(-100), loss 已经是 0，再乘以 weight 保持 0
        # 对于需要计算的位置，乘以对应权重
        weighted_losses = losses * flat_weights
        
        # 计算实际参与 loss 计算的 token 数量和加权后的平均 loss
        # 注意：需要使用 flat_weights 来统计实际计算的权重和 (而不是简单 count)
        total_weight = flat_weights.sum()
        
        if total_weight > 0:
            loss = weighted_losses.sum() / total_weight
        else:
            # 如果没有有效 token (极端情况)，返回 0 loss
            loss = weighted_losses.sum()
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """
        重写 evaluate 以在评估后打印 10 条样本的预测结果
        仅显示：raw_input, raw_output, predicted
        """
        # 先执行标准评估
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # 打印选定的样本
        if self.eval_samples_to_print and self.state.global_step >= 0:
            print(f"\n{'='*80}")
            print(f"当前Step {self.state.global_step} - Sample Predictions (Seed=42)")
            print(f"{'='*80}")
            
            self.model.eval()
            device = self.args.device
            
            for idx, sample in enumerate(self.eval_samples_to_print):
                raw_input = sample["raw_input"]
                raw_output = sample["raw_output"]
                gen_prompt = sample["generation_prompt"]
                
                # Tokenize 用于生成的完整 prompt（含 instruction）
                inputs = self.tokenizer(
                    gen_prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=self.args.max_seq_length
                ).to(device)
                
                # Generate
                with torch.no_grad():
                    generate_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                # 解码生成结果（去掉 input prompt 部分）
                input_length = inputs["input_ids"].shape[1]
                generated_tokens = generate_ids[0][input_length:]
                pred_text = self.tokenizer.decode(
                    generated_tokens, 
                    skip_special_tokens=True
                )
                
                # 打印三个字段，不包含 instruction
                print(f"\n[Sample {idx+1}/{len(self.eval_samples_to_print)}]")
                print(f"Input:    {raw_input[:300]}{'...' if len(raw_input) > 300 else ''}")
                print(f"Output:   {raw_output[:300]}{'...' if len(raw_output) > 300 else ''}")
                print(f"Predict:  {pred_text[:300]}{'...' if len(pred_text) > 300 else ''}")
                print("-" * 80)
            
            print(f"{'='*80}\n")
            
        return output


def is_recommendation_task(instruction: str) -> bool:
    """判断是否为 sid seq -> sid 推荐任务（需要特殊处理 train_on_inputs）"""
    # You are a professional search recommendation engine. Please predict the semantic id of the next item to be visited based on the user's historical visit records. The following are the historical records:
    # Please restore the item information based on the given semantic id.
    # Please generate the corresponding semantic id based on the given item information
    rec_keywords = [
        "search recommendation", 
        "predict the semantic id",
        "recommendation engine",
        "next item"
    ]
    return any(kw in instruction.lower() for kw in rec_keywords)


def generate_and_tokenize_prompt(data_point: Dict, tokenizer, config: TrainingConfig, cutoff_len: int):
    """
    生成 prompt 并 tokenize，同时处理：
    1. 区分 input/output boundary 用于 mask
    2. 仅对推荐任务应用 train_on_inputs 和 output_token_weights
    3. SID 理解任务始终走传统 SFT (所有 token 权重为 1)
    """
    instruction = data_point["instruction"]
    input_text = data_point.get("input", "")
    output = data_point.get("output", "")
    
    # 判断任务类型
    is_rec_task = is_recommendation_task(instruction)
    
    # 构建完整 prompt
    full_prompt = generate_qwen_prompt(data_point)
    
    # Tokenize 完整序列
    full_tokenized = tokenizer(
        full_prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
        add_special_tokens=True,
    )
    
    input_ids = full_tokenized["input_ids"]
    attention_mask = full_tokenized["attention_mask"]
    
    # 添加 EOS token (如果空间足够)
    eos_id = tokenizer.eos_token_id
    if len(input_ids) < cutoff_len and input_ids[-1] != eos_id:
        input_ids.append(eos_id)
        attention_mask.append(1)
    
    labels = input_ids.copy()
    
    # 初始化权重：默认为 1.0 (所有位置都计算 loss)
    loss_weights = [1.0] * len(input_ids)
    
    # 仅对推荐任务应用特殊逻辑
    if is_rec_task:
        if not config.train_on_inputs:
            # 构建只有 input 的 prompt (assistant 部分为空)
            input_only_prompt = generate_qwen_prompt({**data_point, "output": ""})
            input_tokenized = tokenizer(
                input_only_prompt,
                truncation=True,
                max_length=cutoff_len,
                padding=False,
                return_tensors=None,
            )
            input_len = len(input_tokenized["input_ids"])
            
            # Input 部分的 label 设为 -100 (忽略)，权重设为 0
            for i in range(min(input_len, len(labels))):
                labels[i] = -100
                loss_weights[i] = 0.0
            
            # 处理 output 部分权重
            if config.output_token_weights:
                # 从 input_len 开始是 output 部分 (包含生成的 content)
                output_start = input_len
                weight_idx = 0
                
                for pos in range(output_start, len(labels)):
                    if labels[pos] == -100:
                        continue
                        
                    if weight_idx < len(config.output_token_weights):
                        # 赋予指定权重
                        loss_weights[pos] = config.output_token_weights[weight_idx]
                        weight_idx += 1
                    else:
                        # 超出 list 长度的部分，权重设为 0 (忽略这些 token)
                        loss_weights[pos] = 0.0
                    
                    # 遇到 EOS token 特殊处理：权重设为 list[1] (按需求)
                    if input_ids[pos] == eos_id and len(config.output_token_weights) > 1:
                        loss_weights[pos] = config.output_token_weights[1]
                        break
    
    # 对于 SID 理解任务 (item info):
    # - 完全忽略 train_on_inputs (始终为 True 的效果，计算全部 loss)
    # - 完全忽略 output_token_weights (所有 token 权重为 1)
    # - labels 保持与 input_ids 一致 (没有 -100)
    # - loss_weights 保持全 1 (上面已初始化)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "loss_weights": loss_weights,
    }



def prepare_eval_samples(raw_dataset: Dataset, tokenizer, num_samples: int = 5) -> List[Dict]:
    """
    从验证集中随机选取固定样本用于打印 (seed=42)
    保存原始 input/output 用于展示，同时保存生成用的完整 prompt
    """
    # 固定种子选择
    indices = list(range(len(raw_dataset)))
    rng = random.Random(42)
    rng.shuffle(indices)
    selected_indices = indices[:num_samples]
    
    samples = []
    for idx in selected_indices:
        item = raw_dataset[idx]
        raw_input = item.get("input", "")
        raw_output = item.get("output", "")
        
        # 构建用于生成的完整 prompt（包含 instruction，这是模型需要看到的上下文）
        input_prompt_for_generation = generate_qwen_prompt({**item, "output": ""})
        
        samples.append({
            "raw_input": raw_input,      # 仅 input 字段，如 "sid:<a_159>..."
            "raw_output": raw_output,    # 仅 output 字段
            "generation_prompt": input_prompt_for_generation,  # 用于 model.generate()
        })
    
    return samples


def verify_tokenization_and_mask(raw_ds, tokenized_ds, tokenizer, config, num_samples=20):
    """
    验证每种任务类型的 tokenization 和 loss mask 逻辑
    打印：
    - Input 部分后 20 个 token ids 及对应 text/mask/weight
    - Output 部分前 10 个 token ids 及对应 text/mask/weight
    """
    # 分类收集样本索引
    rec_indices = []
    non_rec_indices = []
    
    for i, item in enumerate(raw_ds):
        if len(rec_indices) >= num_samples and len(non_rec_indices) >= num_samples:
            break
        is_rec = is_recommendation_task(item["instruction"])
        if is_rec and len(rec_indices) < num_samples:
            rec_indices.append(i)
        elif not is_rec and len(non_rec_indices) < num_samples:
            non_rec_indices.append(i)
    
    # 验证并打印
    for task_name, indices in [("Rec Task (Recommendation)", rec_indices), 
                               ("Non-Rec Task (Item Info)", non_rec_indices)]:
        if not indices:
            continue
            
        print(f"\n{'='*100}")
        print(f"Verifying {task_name} - First {len(indices)} Samples")
        print(f"Config: train_on_inputs={config.train_on_inputs}, output_token_weights={config.output_token_weights}")
        print(f"{'='*100}")
        
        for rank, idx in enumerate(indices):
            raw_item = raw_ds[idx]
            tokenized_item = tokenized_ds[idx]
            
            # 转为 list（兼容 tensor/numpy/list）
            def to_list(val):
                if isinstance(val, torch.Tensor):
                    return val.cpu().tolist()
                elif isinstance(val, np.ndarray):
                    return val.tolist()
                return val
            
            input_ids = to_list(tokenized_item["input_ids"])
            attention_mask = to_list(tokenized_item["attention_mask"])
            labels = to_list(tokenized_item["labels"])
            loss_weights = to_list(tokenized_item["loss_weights"])
            
            # 计算 input 长度（通过重新 tokenize input-only prompt）
            input_only_prompt = generate_qwen_prompt({**raw_item, "output": ""})
            input_only_tokenized = tokenizer(
                input_only_prompt, 
                truncation=True, 
                max_length=len(input_ids),
                add_special_tokens=True
            )
            input_len = len(input_only_tokenized["input_ids"])
            
            # 截取后 20 个 input tokens 和前 10 个 output tokens
            input_start_idx = max(0, input_len - 20)
            input_slice = slice(input_start_idx, input_len)
            output_slice = slice(input_len, min(input_len + 10, len(input_ids)))
            
            input_ids_seg = input_ids[input_slice]
            output_ids_seg = input_ids[output_slice]
            
            input_mask_seg = attention_mask[input_slice]
            output_mask_seg = attention_mask[output_slice]
            
            input_weights_seg = loss_weights[input_slice]
            output_weights_seg = loss_weights[output_slice]
            
            # 解码为文本
            input_text = tokenizer.decode(input_ids_seg, skip_special_tokens=False)
            output_text = tokenizer.decode(output_ids_seg, skip_special_tokens=False)
            
            # 打印信息
            print(f"\n[Sample {rank+1} | Dataset Index {idx}]")
            print(f"Instruction: {raw_item['instruction'][:80]}...")
            print(f"Raw Input: {raw_item.get('input', '')[:60]}...")
            print(f"Raw Output: {raw_item.get('output', '')[:60]}...")
            
            print(f"\n  Input Part (last {len(input_ids_seg)} tokens, indices {input_start_idx}-{input_len-1}):")
            print(f"    IDs:   {input_ids_seg}")
            print(f"    Text:  '{input_text}'")
            print(f"    Mask:  {input_mask_seg}")
            print(f"    Weight:{input_weights_seg}")
            
            print(f"\n  Output Part (first {len(output_ids_seg)} tokens, indices {input_len}-{input_len+len(output_ids_seg)-1}):")
            print(f"    IDs:   {output_ids_seg}")
            print(f"    Text:  '{output_text}'")
            print(f"    Mask:  {output_mask_seg}")
            print(f"    Weight:{output_weights_seg}")
            
            # 验证逻辑检查
            is_rec = is_recommendation_task(raw_item["instruction"])
            if is_rec and not config.train_on_inputs:
                # Rec task + train_on_inputs=False: input labels 应该为 -100，weight 应该为 0
                if labels[input_start_idx] != -100:
                    print(f"  ⚠️  Warning: Rec task input label is not -100 (got {labels[input_start_idx]})")
                if input_weights_seg[0] != 0.0:
                    print(f"  ⚠️  Warning: Rec task input weight is not 0.0 (got {input_weights_seg[0]})")
            else:
                # Non-rec 或 train_on_inputs=True: input weight 应该为 1.0
                if input_weights_seg[0] != 1.0:
                    print(f"  ⚠️  Warning: Input weight is not 1.0 (got {input_weights_seg[0]})")
            
            print("-" * 100)

def train(
    # model/data params
    base_model: str = "Qwen/Qwen3-0.6B",
    train_data_path: List[str] = None,
    val_data_path: List[str] = None,
    output_dir: str = "./qwen3-training",
    sample: int = -1,
    seed: int = 42,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 12,
    num_epochs: int = 10,
    learning_rate: float = 1e-5,
    cutoff_len: int = 512,
    # custom loss params
    train_on_inputs: bool = False,  # 只对推荐任务生效
    output_token_weights: Optional[List[float]] = None,  # eg: [1.0, 1.0, 1.0]
    # llm hyperparams
    resume_from_checkpoint: str = None,
    gradient_checkpointing: bool = True,
    # eval params
    eval_samples_print_num: int = 10,
):
    """
    主要训练函数
    
    Args:
        train_on_inputs: 是否计算 input 部分的 loss (仅对推荐任务生效)
        output_token_weights: 对 output token 的加权列表
        eval_samples_print_num: eval 时打印的样本数
    """
    assert train_data_path is not None and len(train_data_path) > 0, "必须提供训练数据路径"
    if val_data_path is None:
        val_data_path = []
    
    if resume_from_checkpoint in ["", "None", "none", "XXX"]:
        resume_from_checkpoint = None
    
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    # 设置设备与精度
    device_map = "auto"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"使用 dtype: {dtype}")
    
    # ==================== 模型加载 (Full Fine-tuning) ====================
    print("=== 加载模型 (Full Fine-tuning) ===")
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
        pad_token='<|endoftext|>'  # Qwen3 默认 eos 也是 endoftext
    )
    tokenizer.padding_side = "right"
    
    # 保存 tokenizer 配置以便后续使用
    tokenizer.save_pretrained(output_dir)
    
    # 全参数微调配置
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
    raw_train_ds = raw_train_ds.shuffle(seed=seed)
    
    # 验证集
    raw_val_ds = None
    if val_data_path:
        all_val_ds = [load_dataset("json", data_files=p)["train"] for p in val_data_path]
        raw_val_ds = concatenate_datasets(all_val_ds).shuffle(seed=seed)
    
    # 配置
    config = TrainingConfig(
        train_on_inputs=train_on_inputs,
        output_token_weights=output_token_weights,
    )
    
    print(f"train_on_inputs: {train_on_inputs}")
    print(f"output_token_weights: {output_token_weights}")
    
    # Tokenize 训练集
    print("Tokenizing train set...")
    train_data = raw_train_ds.map(
        lambda x: generate_and_tokenize_prompt(x, tokenizer, config, cutoff_len),
        remove_columns=raw_train_ds.column_names,
        desc="Tokenizing train",
    )
    
    # 验证集处理
    val_data = None
    eval_samples = []
    if raw_val_ds:
        print("Tokenizing validation set...")
        val_data = raw_val_ds.map(
            lambda x: generate_and_tokenize_prompt(x, tokenizer, config, cutoff_len),
            remove_columns=raw_val_ds.column_names,
            desc="Tokenizing val",
        )
        # 准备用于打印的样本 (使用原始文本，用于 generation)
        eval_samples = prepare_eval_samples(raw_val_ds, tokenizer, eval_samples_print_num)
    
    # 设置格式
    columns_to_keep = ["input_ids", "attention_mask", "labels", "loss_weights"]
    train_data = train_data.remove_columns([c for c in train_data.column_names if c not in columns_to_keep])
    train_data.set_format(type="torch")
    
    if val_data:
        val_data = val_data.remove_columns([c for c in val_data.column_names if c not in columns_to_keep])
        val_data.set_format(type="torch")
    
    # ==================== 验证 Tokenization 和 Mask ====================
    print("\n" + "="*100)
    print("PRE-TRAINING VERIFICATION: Tokenization & Loss Mask Check")
    print("="*100)
    verify_tokenization_and_mask(raw_train_ds, train_data, tokenizer, config, num_samples=10)
    
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
        remove_unused_columns=False,  # 重要：我们需要保留 loss_weights
        weight_decay=0.0,  # 全参微调通常用较小 weight decay
        max_grad_norm=1.0,
        seed=seed,
    )
    
    # 将 cutoff_len 传给 trainer 用于 generation
    training_args.max_seq_length = cutoff_len
    
    # Data Collator: 需要正确处理 loss_weights 的 padding
    class WeightedDataCollator(DataCollatorForSeq2Seq):
        def __call__(self, features):
            # 关键修复：先将 Tensor/numpy 转回 list，避免父类转换时的性能警告
            for feature in features:
                for key in ["input_ids", "attention_mask", "labels"]:
                    if key in feature:
                        val = feature[key]
                        if isinstance(val, torch.Tensor):
                            feature[key] = val.cpu().numpy().tolist()
                        elif isinstance(val, np.ndarray):
                            feature[key] = val.tolist()
            
            # 取出 loss_weights
            loss_weights = [f.pop("loss_weights", None) for f in features]
            
            # 调用父类处理
            batch = super().__call__(features)
            
            # 处理 loss_weights...
            if loss_weights[0] is not None:
                max_len = batch["input_ids"].shape[1]
                padded_weights = []
                
                for w in loss_weights:
                    if isinstance(w, torch.Tensor):
                        w = w.cpu().numpy().tolist()
                    elif isinstance(w, np.ndarray):
                        w = w.tolist()
                    
                    if len(w) < max_len:
                        w = w + [0.0] * (max_len - len(w))
                    else:
                        w = w[:max_len]
                    padded_weights.append(w)
                
                batch["loss_weights"] = torch.tensor(padded_weights, dtype=torch.float32)
            
            return batch
    
    collator = WeightedDataCollator(
        tokenizer,
        padding=True,
        return_tensors="pt",
        pad_to_multiple_of=8
    )
    
    # ==================== 开始训练 ====================
    trainer = WeightedCELossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
        data_collator=collator,
        eval_samples_to_print=eval_samples,
    )
    
    # 训练
    model.config.use_cache = False
    print("=== 开始训练 ===")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # 保存最终模型
    print(f"保存模型到 {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("训练完成！")


def generate_qwen_prompt(data_point: Dict) -> str:
    """
    构建 Qwen3 ChatML 格式的 prompt
    确保与训练时使用的模板一致
    """
    instruction = data_point.get("instruction", "")
    input_text = data_point.get("input", "")
    output = data_point.get("output", "")
    
    # Qwen3 默认 system prompt，可根据需要调整
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
    
    
    
帮我修改如上代码，要求如下，要注意请先给出你的思路，再进行更改，以便我进行调整:
1. 训练数据的组分会随着epoch的进行而衰减，简单来说就是Non-Rec部分初始是100%(其自身数据量的100%)，随着训练进行，会逐步衰减到0，但是前k_1个epoch始终是100%，到第m_1个epoch线性衰减到0，k是一个参数，传进来(注意，Non-Rec部分的loss weight始终就是正常sft，不需要做权重衰减)
2. 同理，适用于Rec任务，但是Rec任务衰减的不是数据量，而是loss weight，初始的k_2个epoch的loss weight全是1，也就是正常SFT，到第m_2个epoch，loss weight变为input部分全为0，output部分服从传入的参数，也是线性变成的，
    同时，不是一过了k_1个epoch就直接input全部mask掉，而是input部分的weight也是逐步衰减到0，直到完全到了m_2个epoch才是input部分直接-100
3. eval的时候，valid data set的验证逻辑改变一下，valid set的loss只按input全为0，output服从传入参数的计算方式进行计算，同时，按照output的ID位置计算hit rate，既第一个token的hit rate(这里的hit rate表示所有valid跑完之后第一个token完全正确的比率) , 
    前2个token的hit rate(也就是前两个token完全正确的比率)，前三个token的hit rate，只计算前3个即可