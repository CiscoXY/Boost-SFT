import os
import torch
from llamafactory.train.tuner import run_exp

LOCAL_MODEL_PATH = "/mnt/models/Books/checkpoint-110000" 
OUTPUT_DIR = "/home/dpo_full_stage1"
TENSORBOARD_LOGGING_DIR = "/mnt/llm_tensorboard/DPO_Books_full_stage1_12_30"
#TENSORBOARD_LOGGING_DIR = "/mnt/llm_tensorboard/mix_Movies_12_02"
#TENSORBOARD_LOGGING_DIR = "/mnt/llm_tensorboard/mix_Toys_and_Games_sample50_12_09"
DATASET = "Books_dpo_stage1"
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
        "finetuning_type": "full",  
        
        "pref_beta": 0.1,               
        "pref_ftx": 0.25,               
        
        # === 数据设置 ===
        "dataset": DATASET,
        "template": "qwen",
        "cutoff_len": 512,           
        
        # === 输出设置 ===
        "output_dir": OUTPUT_DIR,
        "overwrite_output_dir": True,
        
        # === 3. 显存优化建议 ===
        "per_device_train_batch_size": 16,    # 全量微调建议先调小，防止 OOM
        "gradient_accumulation_steps": 8,   
        "learning_rate": 5e-7,               # 全量微调学习率通常更低
        "num_train_epochs": 20.0,             # 全量微调收敛快，通常 3-5 epoch 即可，20次太多了
        "lr_scheduler_type": "cosine",
        
        "bf16": True if torch.cuda.is_bf16_supported() else False, 
        "fp16": False,
        "gradient_checkpointing": True,      # 必须开启
        
        # === 4. 推荐添加 DeepSpeed 配置 ===
        # "deepspeed": "examples/deepspeed/ds_z3_config.json", 
        
        "plot_loss": True,
        "logging_dir": TENSORBOARD_LOGGING_DIR,
        "report_to": "tensorboard",
        "logging_steps": 3,
        "save_steps": 1000,
        "save_total_limit": 20,
    }

    print("DPO start...")
    run_exp(args)
    print(f"训练完成！模型权重已保存至 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
