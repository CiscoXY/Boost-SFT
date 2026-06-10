import os
import torch
from llamafactory.train.tuner import run_exp

LOCAL_MODEL_PATH = "/mnt/models/Books/checkpoint-110000" 
OUTPUT_DIR = "/home/dpo_stage0"
TENSORBOARD_LOGGING_DIR = "/mnt/llm_tensorboard/DPO_Books_stage0_12_26"
#TENSORBOARD_LOGGING_DIR = "/mnt/llm_tensorboard/mix_Movies_12_02"
#TENSORBOARD_LOGGING_DIR = "/mnt/llm_tensorboard/mix_Toys_and_Games_sample50_12_09"
DATASET = "Books_dpo_stage0"
#DATASET = "mix_Movies" # mix_Books  mix_Toys_and_Games
#DATASET = "mix_Toys_and_Games"
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"创建目录: {OUTPUT_DIR}")
    if not os.path.exists(TENSORBOARD_LOGGING_DIR):
        os.makedirs(TENSORBOARD_LOGGING_DIR, exist_ok=True)
        print(f"创建目录: {TENSORBOARD_LOGGING_DIR}")
    args = {
        "model_name_or_path": LOCAL_MODEL_PATH,
        "trust_remote_code": True,
        
        "stage": "dpo",            
        "do_train": True,
        "finetuning_type": "lora",  
        
        # === LoRA ===
        "lora_target": "all",         
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        
        "pref_beta": 0.1,              # DPO 的 KL 惩罚系数，通常在 0.1 到 0.5 之间
        "pref_ftx": 0.25,               # SFT 损失
        
        # === 数据设置 ===
        "dataset": DATASET, # 对应 dataset_info.json 中的 key
        "template": "qwen",
        "cutoff_len": 1024,           
        
        # === 输出设置 ===
        "output_dir": OUTPUT_DIR,
        "overwrite_output_dir": True,
        
        "per_device_train_batch_size": 24,   
        "gradient_accumulation_steps": 2,    
        "learning_rate": 5e-6,               # LoRA DPO 
        "num_train_epochs": 20.0,             
        "lr_scheduler_type": "cosine",
        
        "bf16": True if torch.cuda.is_bf16_supported() else False, 
        "fp16": False,
        "gradient_checkpointing": True,
        
        "plot_loss": True,
        "logging_dir": TENSORBOARD_LOGGING_DIR,
        "report_to": "tensorboard",
        "logging_steps": 5,
        "save_steps": 4000,
        "save_total_limit": 50,
    }

    print("DPO start...")
    run_exp(args)
    print(f"训练完成！模型权重已保存至 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
