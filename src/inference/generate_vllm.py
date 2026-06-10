import json
import torch
import os
import glob
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
from typing import List, Dict, Any

# --- [ config ] ---
base_config = {
    "load_all_model" : False,  # 如果为True，则对 model_dir 下所有 '*checkpoint-*' 文件夹推理。如果为False，只推理 model_names 中指定的。
    "model_dir" : "/mnt/models/Books/mix_sft_loss_version" , # /mnt/models/Books/dpo_full_stage1
    "model_names": ["checkpoint-14400" , "checkpoint-14800" , "checkpoint-15200" , "checkpoint-15600" , "checkpoint-16000"], #, "checkpoint-40000" , "checkpoint-50000" , "checkpoint-60000" , "checkpoint-70000" , "checkpoint-80000" , "checkpoint-90000" , "checkpoint-96930"] , 
    "input_file_path" : "/mnt/data/userseq/test/Books_llm.jsonl" , # 待 infer 的文件路径
    "output_dir" : "/mnt/data/generate/Books/sft_loss_version_tp_06_n_20" , # infer 产生的文件保存 dir
    "output_file_suffix" : "tp_06_n_20" # 最后保存的文件路径为 f"{output_dir}/{model_name}_{output_file_suffix}.jsonl"
}

samplingparams = {
    "temperature": 0.6,  # 采样温度
    "top_p": 0.90,       # Top-p 采样
    "top_k": -1,         # Top-k 采样 (-1表示禁用)
    "max_new_tokens": 128, # 最大生成的新 Token 数
    "n": 20,              # vLLM的 'n' 参数，即每个 prompt 重复生成次数
}

# vLLM 硬件配置
MAX_MODEL_LEN = 512
TENSOR_PARALLEL_SIZE = 1 # 使用的 GPU 数量
DTYPE = "bfloat16"       # 模型加载精度 (如果不支持bf16，请改为 "half")


def load_data(data_path: str) -> List[Dict[str, Any]]:
    """加载 JSON Lines 格式的数据集，并返回原始数据列表。"""
    data = []
    if not os.path.exists(data_path):
        print(f"Error: Input file not found at {data_path}")
        return data
        
    print(f"Loading raw data from: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if "instruction" not in item:
                    print(f"Warning: Skipped line due to missing 'instruction': {line.strip()[:50]}...")
                    continue
                if "input" not in item:
                     item["input"] = ""
                
                data.append(item)
            except json.JSONDecodeError:
                print(f"Error decoding JSON line: {line.strip()[:50]}...")
    return data

def apply_qwen_template(tokenizer: AutoTokenizer, item: Dict[str, Any]) -> str:
    """
    根据 LlamaFactory 的 Qwen 模板规则拼接 instruction/input 并应用 Qwen 对话格式。
    """
    instruction = item["instruction"]
    input_text = item["input"]
    
    # LlamaFactory 的拼接逻辑: instruction + \n + input
    if input_text:
        user_content = f"{instruction}\n{input_text}"
    else:
        user_content = instruction
        
    messages = [
        {"role": "user", "content": user_content}
    ]
    
    # Qwen 模板 add_generation_prompt=True 确保添加了 <|im_start|>assistant\n
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    return formatted_prompt


# --- [ infer ] ---

def infer_single_model(
    model_path: str, 
    model_name: str, 
    raw_data: List[Dict[str, Any]], 
    templated_prompts: List[str]
):
    """对单个模型进行推理并保存结果。"""
    
    print("-" * 50)
    print(f"Starting inference for model: {model_name}")
    
    # 加载分词器 (在循环内加载是为了保证模型切换时的正确性，虽然 Qwen 分词器可能不变)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer for {model_name}: {e}. Skipping.")
        return
    
    # 初始化 vLLM
    print(f"Initializing vLLM for {model_name}...")
    try:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            trust_remote_code=True,
            dtype=DTYPE,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=128,
            gpu_memory_utilization=0.85
        )
    except Exception as e:
        print(f"Error initializing vLLM for {model_name}: {e}. Skipping.")
        return

    # 配置采样参数
    sampling_params_obj = SamplingParams(
        n=samplingparams["n"],
        temperature=samplingparams["temperature"],
        top_p=samplingparams["top_p"],
        top_k=samplingparams["top_k"] if samplingparams["top_k"] > 0 else -1,
        max_tokens=samplingparams["max_new_tokens"],
        stop_token_ids=[tokenizer.eos_token_id]
    )

    # 推理
    print(f"Generating predictions (k={samplingparams['n']} repeats per prompt)...")
    outputs = llm.generate(templated_prompts, sampling_params_obj)

    # 整理和保存结果
    final_results = []
    
    for i, raw_item in enumerate(raw_data):
        output = outputs[i]
        predictions = [
            gen.text.strip() for gen in output.outputs
        ]
        
        result_item = raw_item.copy()
        result_item["predict"] = predictions
        final_results.append(result_item)
        
    # 构建输出路径
    output_filename = f"{model_name}_{base_config['output_file_suffix']}.jsonl"
    output_file_path = os.path.join(base_config["output_dir"], output_filename)
    
    print(f"Saving {len(final_results)} results to: {output_file_path}")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in final_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Inference for {model_name} completed.")


def run_vllm_inference_batch():
    """管理模型加载和批量推理的主函数。"""
    
    # 确保输出目录存在
    if not os.path.exists(base_config["output_dir"]):
        os.makedirs(base_config["output_dir"])
        print(f"Created output directory: {base_config['output_dir']}")

    # 1. 加载数据
    raw_data = load_data(base_config["input_file_path"])
    if not raw_data:
        return
        
    # 2. 确定模型列表
    models_to_infer: List[str] = []
    if base_config["load_all_model"]:
        # 查找 model_dir 下所有包含 'checkpoint-' 的文件夹
        search_path_1 = os.path.join(base_config["model_dir"], "*checkpoint-*")
        search_path_2 = os.path.join(base_config["model_dir"], "*ckpt_*")
        models_to_infer = [os.path.basename(d) for d in glob.glob(search_path_1) if os.path.isdir(d)]
        models_to_infer = models_to_infer + [os.path.basename(d) for d in glob.glob(search_path_2) if os.path.isdir(d)]
    else:
        # 使用 model_names 中指定的模型名称
        models_to_infer = base_config["model_names"]

    if not models_to_infer:
        print("Error: No models found or specified for inference. Exiting.")
        return

    print(f"Found {len(models_to_infer)} models to process: {models_to_infer}")
    
    # 预先模板化 Prompts (所有模型使用同一套 Prompt)
    # 必须先加载任意一个 Qwen 分词器来获取模板，这里假设 Qwen 模板在所有 checkpoints 中一致。
    temp_model_path = os.path.join(base_config["model_dir"], models_to_infer[0])
    print(f"Loading tokenizer from base directory {temp_model_path} for template application...")
    try:
        # 尝试从模型基目录加载分词器，因为模板一般在微调过程中不会改变
        temp_tokenizer = AutoTokenizer.from_pretrained(temp_model_path ,  trust_remote_code=True)
    except Exception as e:
        print(f"Error loading base tokenizer for template: {e}. Cannot proceed.")
        return

    print("Applying Qwen chat template to raw data...")
    templated_prompts = [
        apply_qwen_template(temp_tokenizer, item) for item in raw_data
    ]
    
    # 循环推理每个模型
    for model_name in models_to_infer:
        full_model_path = os.path.join(base_config["model_dir"], model_name)
        print(f"推理{full_model_path}中 ==========")
        # 检查路径是否存在且是目录
        if not os.path.isdir(full_model_path):
            print(f"Warning: Model path not found at {full_model_path}. Skipping {model_name}.")
            continue
            
        # 调用单模型推理函数
        infer_single_model(full_model_path, model_name, raw_data, templated_prompts)
        
    print("\nBatch inference finished for all models.")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. vLLM requires a GPU to run.")
    else:
        run_vllm_inference_batch()