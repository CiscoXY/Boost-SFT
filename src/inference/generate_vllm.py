import json
import torch
import os
import glob
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
from typing import List, Dict, Any

base_config = {
    "load_all_model" : False,
    "model_dir" : "/path/to/models/experiment",
    "model_names": ["checkpoint-XXXXX", "checkpoint-YYYYY"],
    "input_file_path" : "/path/to/data/test/dataset_llm.jsonl",
    "output_dir" : "/path/to/generated/experiment_output",
    "output_file_suffix" : "experiment_name"
}

samplingparams = {
    "temperature": 0.6,
    "top_p": 0.90,
    "top_k": -1,
    "max_new_tokens": 128,
    "n": 20,
}

MAX_MODEL_LEN = 512
TENSOR_PARALLEL_SIZE = 1
DTYPE = "bfloat16"


def load_data(data_path: str) -> List[Dict[str, Any]]:
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
    instruction = item["instruction"]
    input_text = item["input"]

    if input_text:
        user_content = f"{instruction}\n{input_text}"
    else:
        user_content = instruction

    messages = [
        {"role": "user", "content": user_content}
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return formatted_prompt


def infer_single_model(
    model_path: str,
    model_name: str,
    raw_data: List[Dict[str, Any]],
    templated_prompts: List[str]
):
    print("-" * 50)
    print(f"Starting inference for model: {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer for {model_name}: {e}. Skipping.")
        return

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

    sampling_params_obj = SamplingParams(
        n=samplingparams["n"],
        temperature=samplingparams["temperature"],
        top_p=samplingparams["top_p"],
        top_k=samplingparams["top_k"] if samplingparams["top_k"] > 0 else -1,
        max_tokens=samplingparams["max_new_tokens"],
        stop_token_ids=[tokenizer.eos_token_id]
    )

    print(f"Generating predictions (k={samplingparams['n']} repeats per prompt)...")
    outputs = llm.generate(templated_prompts, sampling_params_obj)

    final_results = []

    for i, raw_item in enumerate(raw_data):
        output = outputs[i]
        predictions = [
            gen.text.strip() for gen in output.outputs
        ]

        result_item = raw_item.copy()
        result_item["predict"] = predictions
        final_results.append(result_item)

    output_filename = f"{model_name}_{base_config['output_file_suffix']}.jsonl"
    output_file_path = os.path.join(base_config["output_dir"], output_filename)

    print(f"Saving {len(final_results)} results to: {output_file_path}")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in final_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Inference for {model_name} completed.")


def run_vllm_inference_batch():
    if not os.path.exists(base_config["output_dir"]):
        os.makedirs(base_config["output_dir"])
        print(f"Created output directory: {base_config['output_dir']}")

    raw_data = load_data(base_config["input_file_path"])
    if not raw_data:
        return

    models_to_infer: List[str] = []
    if base_config["load_all_model"]:
        search_path_1 = os.path.join(base_config["model_dir"], "*checkpoint-*")
        search_path_2 = os.path.join(base_config["model_dir"], "*ckpt_*")
        models_to_infer = [os.path.basename(d) for d in glob.glob(search_path_1) if os.path.isdir(d)]
        models_to_infer = models_to_infer + [os.path.basename(d) for d in glob.glob(search_path_2) if os.path.isdir(d)]
    else:
        models_to_infer = base_config["model_names"]

    if not models_to_infer:
        print("Error: No models found or specified for inference. Exiting.")
        return

    print(f"Found {len(models_to_infer)} models to process: {models_to_infer}")

    temp_model_path = os.path.join(base_config["model_dir"], models_to_infer[0])
    print(f"Loading tokenizer from base directory {temp_model_path} for template application...")
    try:
        temp_tokenizer = AutoTokenizer.from_pretrained(temp_model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading base tokenizer for template: {e}. Cannot proceed.")
        return

    print("Applying Qwen chat template to raw data...")
    templated_prompts = [
        apply_qwen_template(temp_tokenizer, item) for item in raw_data
    ]

    for model_name in models_to_infer:
        full_model_path = os.path.join(base_config["model_dir"], model_name)
        print(f"Inferencing {full_model_path} ==========")
        if not os.path.isdir(full_model_path):
            print(f"Warning: Model path not found at {full_model_path}. Skipping {model_name}.")
            continue

        infer_single_model(full_model_path, model_name, raw_data, templated_prompts)

    print("\nBatch inference finished for all models.")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. vLLM requires a GPU to run.")
    else:
        run_vllm_inference_batch()