import os
import torch
from llamafactory.train.tuner import run_exp

LOCAL_MODEL_PATH = "/mnt/models/Books-checkpoint-10000" 
LOCAL_MODEL_PATH = "/mnt/Qwen3-0.6B-addtoken"
OUTPUT_DIR = "/home/sft"
TENSORBOARD_LOGGING_DIR = "/mnt/llm_tensorboard/mix_Books_12_02"
#TENSORBOARD_LOGGING_DIR = "/mnt/llm_tensorboard/mix_Movies_12_02"
#TENSORBOARD_LOGGING_DIR = "/mnt/llm_tensorboard/mix_Toys_and_Games_sample50_12_09"
DATASET = "mix_Books"
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
        # === 模型加载 ===
        "model_name_or_path": LOCAL_MODEL_PATH,
        "trust_remote_code": True,
        
        # === 核心阶段设置 ===
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "full",   # <--- 全量微调
        
        # === 数据设置 ===
        "dataset": DATASET, # dataset_info.json 中注册的key    "mix_Books" , "pipeline_test"
        "template": "qwen",             # 必须是 qwen 模板
        "cutoff_len": 1024,
        "train_on_input":True,
        
        # === 输出设置 ===
        "output_dir": OUTPUT_DIR,
        "overwrite_output_dir": True,
        
        # === 训练超参数 (针对全 GPU 内存优化) ===
        "per_device_train_batch_size": 12,    # 假设小模型可以容纳更大的 Batch Size
        "gradient_accumulation_steps": 3,    # 梯度累积设置为 4，总 Batch Size 为 16
        "learning_rate": 5e-6,               # 全量微调学习率继续保持在较低水平
        "num_train_epochs": 30.0,
        
        # === 显存与精度优化 ===
        # 推荐使用 bf16 以获得更好的数值稳定性，如果不支持则使用 fp16
        "bf16": True if torch.cuda.is_bf16_supported() else False, 
        "fp16": False if torch.cuda.is_bf16_supported() else True, 
        "gradient_checkpointing": True,      # 仍然建议开启，作为额外的显存保护
        # 注意：此处不再需要 DeepSpeed 或 FSDP 配置
        
        # === 日志与保存 ===
        "plot_loss": True,
        "logging_dir": TENSORBOARD_LOGGING_DIR,
        "report_to":"tensorboard",
        "logging_steps": 5,
        "save_steps": 10000,
        "save_total_limit": 50,
        "save_only_model": False,
    }

    print("SFT start...")
    run_exp(args)
    print(f"训练完成！模型权重已保存至 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
    