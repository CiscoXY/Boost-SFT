import os
import torch
from llamafactory.train.tuner import run_exp

LOCAL_MODEL_PATH = "/path/to/models/checkpoint-XXXXX"
OUTPUT_DIR = "/path/to/dpo_full_output"
TENSORBOARD_LOGGING_DIR = "/path/to/tensorboard/DPO_full_experiment"
DATASET = "example_dpo_dataset"
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Created directory: {OUTPUT_DIR}")
    if not os.path.exists(TENSORBOARD_LOGGING_DIR):
        os.makedirs(TENSORBOARD_LOGGING_DIR, exist_ok=True)
        print(f"Created directory: {TENSORBOARD_LOGGING_DIR}")
    args = {
        "model_name_or_path": LOCAL_MODEL_PATH,
        "trust_remote_code": True,
        
        "stage": "dpo",            
        "do_train": True,
        "finetuning_type": "full",  
        
        "pref_beta": 0.1,
        "pref_ftx": 0.25,

        "dataset": DATASET,
        "template": "qwen",
        "cutoff_len": 512,

        "output_dir": OUTPUT_DIR,
        "overwrite_output_dir": True,

        "per_device_train_batch_size": 16,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5e-7,
        "num_train_epochs": 20.0,
        "lr_scheduler_type": "cosine",

        "bf16": True if torch.cuda.is_bf16_supported() else False,
        "fp16": False,
        "gradient_checkpointing": True,

        "plot_loss": True,
        "logging_dir": TENSORBOARD_LOGGING_DIR,
        "report_to": "tensorboard",
        "logging_steps": 3,
        "save_steps": 1000,
        "save_total_limit": 20,
    }

    print("DPO start...")
    run_exp(args)
    print(f"Training complete! Model weights saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
