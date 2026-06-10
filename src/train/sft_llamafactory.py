import os
import torch
from llamafactory.train.tuner import run_exp

LOCAL_MODEL_PATH = "/path/to/models/checkpoint-XXXXX"
OUTPUT_DIR = "/path/to/sft_output"
TENSORBOARD_LOGGING_DIR = "/path/to/tensorboard/example_experiment"
DATASET = "example_dataset"
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

        "stage": "sft",
        "do_train": True,
        "finetuning_type": "full",

        "dataset": DATASET,
        "template": "qwen",
        "cutoff_len": 1024,
        "train_on_input": True,

        "output_dir": OUTPUT_DIR,
        "overwrite_output_dir": True,

        "per_device_train_batch_size": 12,
        "gradient_accumulation_steps": 3,
        "learning_rate": 5e-6,
        "num_train_epochs": 30.0,

        "bf16": True if torch.cuda.is_bf16_supported() else False,
        "fp16": False if torch.cuda.is_bf16_supported() else True,
        "gradient_checkpointing": True,

        "plot_loss": True,
        "logging_dir": TENSORBOARD_LOGGING_DIR,
        "report_to": "tensorboard",
        "logging_steps": 5,
        "save_steps": 10000,
        "save_total_limit": 50,
        "save_only_model": False,
    }

    print("SFT start...")
    run_exp(args)
    print(f"Training complete! Model weights saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
    