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
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)


@dataclass
class TrainingConfig:
    train_on_inputs: bool = False
    output_token_weights: Optional[List[float]] = None
    is_recommendation_task: bool = False


class WeightedCELossTrainer(Trainer):
    def __init__(self, *args, eval_samples_to_print: Optional[List[Dict]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_samples_to_print = eval_samples_to_print or []
        self.global_step_tracker = 0
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        loss_weights = inputs.pop("loss_weights", None)

        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        if loss_weights is not None:
            shift_weights = loss_weights[..., 1:].contiguous()
        else:
            shift_weights = (shift_labels != -100).float()

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
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        if self.eval_samples_to_print and self.state.global_step >= 0:
            print(f"\n{'='*80}")
            print(f"Current Step {self.state.global_step} - Sample Predictions (Seed=42)")
            print(f"{'='*80}")

            self.model.eval()
            device = self.args.device

            for idx, sample in enumerate(self.eval_samples_to_print):
                raw_input = sample["raw_input"]
                raw_output = sample["raw_output"]
                gen_prompt = sample["generation_prompt"]

                inputs = self.tokenizer(
                    gen_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.args.max_seq_length
                ).to(device)

                with torch.no_grad():
                    generate_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                input_length = inputs["input_ids"].shape[1]
                generated_tokens = generate_ids[0][input_length:]
                pred_text = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True
                )

                print(f"\n[Sample {idx+1}/{len(self.eval_samples_to_print)}]")
                print(f"Input:    {raw_input[:300]}{'...' if len(raw_input) > 300 else ''}")
                print(f"Output:   {raw_output[:300]}{'...' if len(raw_output) > 300 else ''}")
                print(f"Predict:  {pred_text[:300]}{'...' if len(pred_text) > 300 else ''}")
                print("-" * 80)

            print(f"{'='*80}\n")

        return output


def is_recommendation_task(instruction: str) -> bool:
    rec_keywords = [
        "search recommendation", 
        "predict the semantic id",
        "recommendation engine",
        "next item"
    ]
    return any(kw in instruction.lower() for kw in rec_keywords)


def generate_and_tokenize_prompt(data_point: Dict, tokenizer, config: TrainingConfig, cutoff_len: int):
    instruction = data_point["instruction"]
    input_text = data_point.get("input", "")
    output = data_point.get("output", "")

    is_rec_task = is_recommendation_task(instruction)

    full_prompt = generate_qwen_prompt(data_point)

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

    eos_id = tokenizer.eos_token_id
    if len(input_ids) < cutoff_len and input_ids[-1] != eos_id:
        input_ids.append(eos_id)
        attention_mask.append(1)

    labels = input_ids.copy()
    loss_weights = [1.0] * len(input_ids)

    if is_rec_task:
        if not config.train_on_inputs:
            input_only_prompt = generate_qwen_prompt({**data_point, "output": ""})
            input_tokenized = tokenizer(
                input_only_prompt,
                truncation=True,
                max_length=cutoff_len,
                padding=False,
                return_tensors=None,
            )
            input_len = len(input_tokenized["input_ids"])

            for i in range(min(input_len, len(labels))):
                labels[i] = -100
                loss_weights[i] = 0.0

            if config.output_token_weights:
                output_start = input_len
                weight_idx = 0

                for pos in range(output_start, len(labels)):
                    if labels[pos] == -100:
                        continue

                    if weight_idx < len(config.output_token_weights):
                        loss_weights[pos] = config.output_token_weights[weight_idx]
                        weight_idx += 1
                    else:
                        loss_weights[pos] = 0.0

                    if input_ids[pos] == eos_id and len(config.output_token_weights) > 1:
                        loss_weights[pos] = config.output_token_weights[1]
                        break

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "loss_weights": loss_weights,
    }

def prepare_eval_samples(raw_dataset: Dataset, tokenizer, num_samples: int = 5) -> List[Dict]:
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


def verify_tokenization_and_mask(raw_ds, tokenized_ds, tokenizer, config, num_samples=20):
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

            input_only_prompt = generate_qwen_prompt({**raw_item, "output": ""})
            input_only_tokenized = tokenizer(
                input_only_prompt,
                truncation=True,
                max_length=len(input_ids),
                add_special_tokens=True
            )
            input_len = len(input_only_tokenized["input_ids"])

            input_start_idx = max(0, input_len - 20)
            input_slice = slice(input_start_idx, input_len)
            output_slice = slice(input_len, min(input_len + 10, len(input_ids)))

            input_ids_seg = input_ids[input_slice]
            output_ids_seg = input_ids[output_slice]

            input_mask_seg = attention_mask[input_slice]
            output_mask_seg = attention_mask[output_slice]

            input_weights_seg = loss_weights[input_slice]
            output_weights_seg = loss_weights[output_slice]

            input_text = tokenizer.decode(input_ids_seg, skip_special_tokens=False)
            output_text = tokenizer.decode(output_ids_seg, skip_special_tokens=False)

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

            is_rec = is_recommendation_task(raw_item["instruction"])
            if is_rec and not config.train_on_inputs:
                if labels[input_start_idx] != -100:
                    print(f"  ⚠️  Warning: Rec task input label is not -100 (got {labels[input_start_idx]})")
                if input_weights_seg[0] != 0.0:
                    print(f"  ⚠️  Warning: Rec task input weight is not 0.0 (got {input_weights_seg[0]})")
            else:
                if input_weights_seg[0] != 1.0:
                    print(f"  ⚠️  Warning: Input weight is not 1.0 (got {input_weights_seg[0]})")

            print("-" * 100)

def train(
    base_model: str = "/path/to/base/model",
    train_data_path: List[str] = None,
    val_data_path: List[str] = None,
    output_dir: str = "./output",
    sample: int = -1,
    seed: int = 42,
    batch_size: int = 128,
    micro_batch_size: int = 12,
    num_epochs: int = 10,
    learning_rate: float = 1e-5,
    cutoff_len: int = 512,
    train_on_inputs: bool = False,
    output_token_weights: Optional[List[float]] = None,
    resume_from_checkpoint: str = None,
    gradient_checkpointing: bool = True,
    eval_samples_print_num: int = 10,
):
    assert train_data_path is not None and len(train_data_path) > 0, "Must provide training data path"
    if val_data_path is None:
        val_data_path = []

    if resume_from_checkpoint in ["", "None", "none", "XXX"]:
        resume_from_checkpoint = None

    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using dtype: {dtype}")

    print("=== Loading model (Full Fine-tuning) ===")
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
        print("Enabling Gradient Checkpointing...")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    print("=== Loading datasets ===")
    all_train_ds = [load_dataset("json", data_files=p)["train"] for p in train_data_path]
    raw_train_ds = concatenate_datasets(all_train_ds)
    if sample > -1:
        raw_train_ds = raw_train_ds.select(range(min(sample, len(raw_train_ds))))
    raw_train_ds = raw_train_ds.shuffle(seed=seed)

    raw_val_ds = None
    if val_data_path:
        all_val_ds = [load_dataset("json", data_files=p)["train"] for p in val_data_path]
        raw_val_ds = concatenate_datasets(all_val_ds).shuffle(seed=seed)

    config = TrainingConfig(
        train_on_inputs=train_on_inputs,
        output_token_weights=output_token_weights,
    )

    print(f"train_on_inputs: {train_on_inputs}")
    print(f"output_token_weights: {output_token_weights}")

    print("Tokenizing train set...")
    train_data = raw_train_ds.map(
        lambda x: generate_and_tokenize_prompt(x, tokenizer, config, cutoff_len),
        remove_columns=raw_train_ds.column_names,
        desc="Tokenizing train",
    )

    val_data = None
    eval_samples = []
    if raw_val_ds:
        print("Tokenizing validation set...")
        val_data = raw_val_ds.map(
            lambda x: generate_and_tokenize_prompt(x, tokenizer, config, cutoff_len),
            remove_columns=raw_val_ds.column_names,
            desc="Tokenizing val",
        )
        eval_samples = prepare_eval_samples(raw_val_ds, tokenizer, eval_samples_print_num)

    columns_to_keep = ["input_ids", "attention_mask", "labels", "loss_weights"]
    train_data = train_data.remove_columns([c for c in train_data.column_names if c not in columns_to_keep])
    train_data.set_format(type="torch")

    if val_data:
        val_data = val_data.remove_columns([c for c in val_data.column_names if c not in columns_to_keep])
        val_data.set_format(type="torch")

    print("\n" + "="*100)
    print("PRE-TRAINING VERIFICATION: Tokenization & Loss Mask Check")
    print("="*100)
    verify_tokenization_and_mask(raw_train_ds, train_data, tokenizer, config, num_samples=10)

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
        remove_unused_columns=False,
        weight_decay=0.0,
        max_grad_norm=1.0,
        seed=seed,
    )

    training_args.max_seq_length = cutoff_len

    class WeightedDataCollator(DataCollatorForSeq2Seq):
        def __call__(self, features):
            for feature in features:
                for key in ["input_ids", "attention_mask", "labels"]:
                    if key in feature:
                        val = feature[key]
                        if isinstance(val, torch.Tensor):
                            feature[key] = val.cpu().numpy().tolist()
                        elif isinstance(val, np.ndarray):
                            feature[key] = val.tolist()

            loss_weights = [f.pop("loss_weights", None) for f in features]

            batch = super().__call__(features)

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

    trainer = WeightedCELossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
        data_collator=collator,
        eval_samples_to_print=eval_samples,
    )

    model.config.use_cache = False
    print("=== Starting training ===")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete!")


def generate_qwen_prompt(data_point: Dict) -> str:
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
