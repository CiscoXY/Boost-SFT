import os
import torch
import numpy as np
import json
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from data.esay_dataset import ItemData, TigerDataset , TemplatedTextDataset , TigerDataset_Direct
from accelerate import Accelerator

from data.utils import item_batch_to
from data.utils import cycle
from data.utils import next_item_batch
from data.utils import write_record_log
from modules.rqvae import RqVae
from modules.quantize import QuantizeForwardMode
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.utils import parse_config
from torch.optim import AdamW
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from tqdm import tqdm
from tiger_utils.data_utils import process_rq_vae_codebook
import time
import glob
import re
from pathlib import Path
import gc
from vllm import LLM

def train(
    load_data_dir,
    save_data_dir,
    save_model_dir,
    log_dir,
    record_path,
    embedding_size,
    iterations,
    batch_size,
    learning_rate,
    weight_decay,
    pretrained_rqvae_path,
    use_kmeans_init,
    do_eval,
    gradient_accumulate_every,
    save_model_every,
    eval_every,
    commitment_weight,
    vae_input_dim,
    vae_embed_dim,
    vae_hidden_dims,
    vae_codebook_size,
    vae_codebook_normalize,
    vae_sim_vq,
    vae_n_layers,
):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_data_dir, exist_ok=True)
    os.makedirs(save_model_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    vae_codebook_mode = QuantizeForwardMode.ROTATION_TRICK
    accelerator = Accelerator()
    device = accelerator.device

    raw_dataset_start = time.time()
    print(f"vae_input_dim = {vae_input_dim}")
    raw_dataset = TigerDataset(
        dim_cutoff=vae_input_dim , embedding_size=tuple(embedding_size),data_dir=load_data_dir,record_path=record_path
    )
    print(f"Created raw_dataset in {(time.time() - raw_dataset_start):.2f} s")
    write_record_log(f"Created raw_dataset in {(time.time() - raw_dataset_start):.2f} s" , log_path=record_path)

    train_dataloader_start = time.time()

    train_dataset = ItemData(raw_dataset, train_test_split="train")
    train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size, False)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=None,
        collate_fn=lambda batch: batch,
    )
    train_dataloader = cycle(train_dataloader)
    print(f"Created train_dataloader in {(time.time() - train_dataloader_start):.2f} s")
    write_record_log(f"Created train_dataloader in {(time.time() - train_dataloader_start):.2f} s" , log_path=record_path)


    eval_dataloader_start = time.time()

    eval_dataset = ItemData(raw_dataset, train_test_split="eval")
    eval_sampler = BatchSampler(RandomSampler(eval_dataset), batch_size, False)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=None,
        collate_fn=lambda batch: batch,
    )
    print(f"Created eval_dataloader in {(time.time() - eval_dataloader_start):.2f} s")
    write_record_log(f"Created eval_dataloader in {(time.time() - eval_dataloader_start):.2f} s" , log_path=record_path)

    train_loader_prepare = time.time()

    train_dataloader = accelerator.prepare(train_dataloader)
    print(f"Created train_dataloader_prepare in {(time.time() - train_loader_prepare):.2f} s")
    write_record_log(f"Created train_dataloader_prepare in {(time.time() - train_loader_prepare):.2f} s ", log_path=record_path)

    model = RqVae(
        input_dim=vae_input_dim,
        embed_dim=vae_embed_dim,
        hidden_dims=vae_hidden_dims,
        codebook_size=vae_codebook_size,
        codebook_kmeans_init=use_kmeans_init and pretrained_rqvae_path is None,
        codebook_normalize=vae_codebook_normalize,
        codebook_sim_vq=vae_sim_vq,
        codebook_mode=vae_codebook_mode,
        n_layers=vae_n_layers,
        n_cat_features=0,
        commitment_weight=commitment_weight,
    )
    model_config = {
        "input_dim": vae_input_dim,
        "embed_dim": vae_embed_dim,
        "hidden_dims": vae_hidden_dims,
        "codebook_size": vae_codebook_size,
        "codebook_kmeans_init": False,
        "codebook_normalize": vae_codebook_normalize,
        "codebook_sim_vq": vae_sim_vq,
        "n_layers": vae_n_layers,
        "n_cat_features": 0,
        "commitment_weight": commitment_weight,
    }
    os.makedirs(save_model_dir, exist_ok=True)
    with open(os.path.join(save_model_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, ensure_ascii=False, indent=2)

    optimizer = AdamW(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    start_iter = 0
    if pretrained_rqvae_path is not None:
        model.load_pretrained(pretrained_rqvae_path)
        state = torch.load(
            pretrained_rqvae_path, map_location=device, weights_only=False
        )
        optimizer.load_state_dict(state["optimizer"])
        start_iter = state["iter"] + 1

    model, optimizer = accelerator.prepare(model, optimizer)

    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_feats=0,
        rqvae_weights_path=pretrained_rqvae_path,
        rqvae_codebook_normalize=vae_codebook_normalize,
        rqvae_sim_vq=vae_sim_vq,
    )
    tokenizer.rq_vae = model

    with tqdm(
        initial=start_iter,
        total=start_iter + iterations,
        disable=not accelerator.is_main_process,
    ) as pbar:
        losses = [[], [], []]
        for iter in range(start_iter, start_iter + 1 + iterations):
            model.train()
            total_loss = 0
            t = 0.2
            if iter == 0 and use_kmeans_init:
                kmeans_init_data = item_batch_to(
                    train_dataset[torch.arange(min(40000, len(train_dataset)))], device
                )
                model(kmeans_init_data, t)

            optimizer.zero_grad()
            for _ in range(gradient_accumulate_every):
                data = next_item_batch(train_dataloader, device)
                with accelerator.autocast():
                    model_output = model(data, gumbel_t=t)
                    loss = model_output.loss
                    loss = loss / gradient_accumulate_every
                    total_loss += loss

            accelerator.backward(total_loss)

            losses[0].append(total_loss.cpu().item())
            losses[1].append(model_output.reconstruction_loss.cpu().item())
            losses[2].append(model_output.rqvae_loss.cpu().item())
            losses[0] = losses[0][-1000:]
            losses[1] = losses[1][-1000:]
            losses[2] = losses[2][-1000:]
            if iter % 100 == 0:
                print_loss = np.mean(losses[0])
                print_rec_loss = np.mean(losses[1])
                print_vae_loss = np.mean(losses[2])

                writer.add_scalar('Train/Total_Loss', print_loss, iter)
                writer.add_scalar('Train/Reconstruction_Loss', print_rec_loss, iter)
                writer.add_scalar('Train/RQVAE_Loss', print_vae_loss, iter)

            pbar.set_description(
                f"loss: {print_loss:.4f}, rl: {print_rec_loss:.4f}, vl: {print_vae_loss:.4f}"
            )

            accelerator.wait_for_everyone()

            optimizer.step()

            accelerator.wait_for_everyone()

            if do_eval and ((iter + 1) % eval_every == 0 or iter + 1 == iterations):
                model.eval()
                eval_start = time.time()
                with tqdm(
                    eval_dataloader, desc=f"Eval {iter+1}", disable=True
                ) as pbar_eval:
                    eval_losses = [[], [], []]
                    for batch in pbar_eval:
                        data = item_batch_to(batch, device)
                        with torch.no_grad():
                            eval_model_output = model(data, gumbel_t=t)

                        eval_losses[0].append(eval_model_output.loss.cpu().item())
                        eval_losses[1].append(
                            eval_model_output.reconstruction_loss.cpu().item()
                        )
                        eval_losses[2].append(eval_model_output.rqvae_loss.cpu().item())

                    eval_losses = np.array(eval_losses).mean(axis=-1)
                    print(f"\n{'='*60}")
                    print(f"EVALUATION RESULTS - Iteration {iter+1}")
                    print(f"{'='*60}")
                    print(f"Eval Total Loss:          {eval_losses[0]:.6f}")
                    print(f"Eval Reconstruction Loss: {eval_losses[1]:.6f}")
                    print(f"Eval RQVAE Loss:          {eval_losses[2]:.6f}")
                    writer.add_scalar('Eval/Total_Loss', eval_losses[0], iter+1)
                    writer.add_scalar('Eval/Reconstruction_Loss', eval_losses[1], iter+1)
                    writer.add_scalar('Eval/RQVAE_Loss', eval_losses[2], iter+1)
                eval_end = time.time()
                write_record_log(f"Eval Res - Iter {iter+1} total time {(eval_end - eval_start):.2f} s", log_path=record_path)

            if accelerator.is_main_process:
                if (iter + 1) % save_model_every == 0 or iter + 1 == iterations:
                    state = {
                        "iter": iter,
                        "model": accelerator.get_state_dict(model),
                        "optimizer": optimizer.state_dict(),
                    }
                    torch.save(
                        state, os.path.join(save_model_dir, f"checkpoint_{iter+1}.pt")
                    )
            pbar.update(1)
            
    writer.close()

def generate_emb(
    load_data_dir : str,
    k_files_per_save : int , 
    save_data_dir : str,
    pretrained_rqvae_paths : list[str],
    record_path : str,
    pre_load_embedding_size : list[int],
):
    """
    Generate SIDs from embeddings.

    load_data_dir : directory containing .pt files with sku and embedding fields
    k_files_per_save : batch k pt files into one TigerDataset for inference, reducing memory usage
    save_data_dir : directory to save results
    pretrained_rqvae_paths : list of pretrained RQVAE model paths
    record_path : log file path
    pre_load_embedding_size : pre-allocated tensor size to reduce concatenation overhead
    """
    accelerator = Accelerator()
    device = accelerator.device
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)
    def process_single_model(model_path , full_dataset , save_sign):
        """
        Read config.json from the parent directory of model_path and initialize the model.
        """
        config =  json.load(open(os.path.join(os.path.dirname(model_path) , "config.json"), "r", encoding="utf-8"))
        n_layers = config["n_layers"]
        codebook_size = config["codebook_size"]

        save_data_dir4model = os.path.join(save_data_dir, os.path.splitext(os.path.basename(model_path))[0])
        if not os.path.exists(save_data_dir4model):
            os.makedirs(save_data_dir4model)

        start_time = time.time()
        # Auto-detects config.json from parent directory
        tokenizer = SemanticIdTokenizer(
            rqvae_weights_path=model_path,
            )
        tokenizer.rq_vae = accelerator.prepare(tokenizer.rq_vae)
        write_record_log("================" ,  log_path=record_path)
        write_record_log(f"Loaded {model_path} in {(time.time() - start_time):.2f} s" , log_path=record_path)

        if accelerator.is_main_process:
            tokenizer.reset()
            tokenizer.rq_vae.eval()

            write_record_log(f"{model_path} generate start ......" , log_path=record_path)
            generate_start = time.time()
            corpus_ids , sku_ids = tokenizer.precompute_corpus_ids(full_dataset)
            write_record_log(f"{model_path} precompute_corpus_ids took {(time.time() - generate_start):.3f}s")
            max_duplicates = corpus_ids[:, -1].max() / corpus_ids.shape[0]

            _, counts = torch.unique(
                corpus_ids[:, :], dim=0, return_counts=True
            )
            p = counts / corpus_ids.shape[0]
            rqvae_entropy = -(p * torch.log(p)).sum()

            for sid in range(n_layers):
                _, counts = torch.unique(corpus_ids[:, sid], return_counts=True)
                codebook_usage = len(counts) / codebook_size
                print(
                    f"Codebook Usage Layer {sid}: {codebook_usage:.4f} ({len(counts)}/{codebook_size})"
                )
                write_record_log(f'Codebook_Usage_Layer_{sid} : {codebook_usage:.4f}' , log_path=record_path)
            write_record_log(f"RQVAE Entropy : {(rqvae_entropy.cpu().item()):.6f}" , log_path=record_path)
            write_record_log(f"Max Duplicates : {(max_duplicates.cpu().item()):.6f}" , log_path=record_path)
            write_record_log(f"{model_path} Codebook computation/metrics total time {(time.time() - generate_start):.2f} s", log_path=record_path)
            
            if hasattr(corpus_ids, "cpu"):
                corpus_ids = corpus_ids.cpu().tolist()
            elif hasattr(corpus_ids, "numpy"):
                corpus_ids = corpus_ids.numpy().tolist()

            (
            item_to_tokens,
            tokens_to_item,
            ) = process_rq_vae_codebook(corpus_ids , sku_ids)
            write_record_log(f"{model_path} Codebook generation total time {(time.time() - generate_start):.2f} s", log_path=record_path)
            save_data_dict = {
                "item_to_tokens": item_to_tokens,
                "tokens_to_item": tokens_to_item,
            }
            torch.save(
                save_data_dict, os.path.join(save_data_dir4model, f"saved_tokenizer_data{save_sign}.pt")
            )
            write_record_log(f"{model_path} total time {(time.time() - generate_start):.2f} s", log_path=record_path)
    
    pt_files = glob.glob(str(Path(load_data_dir) / "*.pt"))
    # Batch pt files into TigerDataset to reduce memory usage
    pt_file_batchs = [pt_files[i:i+k_files_per_save] for i in range(0, len(pt_files), k_files_per_save)]
    for idx , pt_file_batch in enumerate(pt_file_batchs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        raw_dataset = TigerDataset(
            pt_files=pt_file_batch, dim_cutoff=pre_load_embedding_size[-1] , embedding_size=tuple(pre_load_embedding_size) , record_path=record_path
        )
        dataset = ItemData(raw_dataset, train_test_split="all")
        for pretrained_rqvae_path in pretrained_rqvae_paths:
            process_single_model(pretrained_rqvae_path , dataset , idx)
    write_record_log("========== GENERATE DONE ==========" , log_path=record_path)

def generate_info(
    load_data_dir : str,
    save_data_dir : str,
    embedding_model_path : str,
    pretrained_rqvae_path : str,
    record_path : str,
):
    """
    End-to-end SID generation from sku info (JSON files containing sku and text template fields) to sid (txt).
    Primarily handles incremental updates, so only one embedding model and one pretrained_rqvae_path are supported.

    load_data_dir : directory containing sku info files (JSON format only)
    save_data_dir : directory to save results as txt, with 7 tab-separated fields: sku, sid1, sid2, sid3, num_sid1, num_sid2, num_sid3
    embedding_model_path : embedding model path
    pretrained_rqvae_path : pretrained RQVAE model path
    record_path : log file path
    """
    accelerator = Accelerator()
    device = accelerator.device
    os.makedirs(save_data_dir, exist_ok=True)

    BATCH_SIZE = 10_0000
    TEXT_TEMPLATE = """
    Product name: {item_name}
    Category L3: {cid1_name} | {cid2_name} | {cid3_name}
    """.strip()
    MAX_LENGTH = 1024
    def TemplatedTextDataset_fn(batch):
        """Batch processing: collect text, sku, and metadata"""
        texts = [item["text"] for item in batch]
        skus = [item["sku"] for item in batch]
        metadata = [
            {k: v for k, v in item.items() if k not in ["text", "sku"]} 
            for item in batch
        ]
        return {"text": texts, "sku": skus, "metadata": metadata}
    

    start_time = time.time()
    # Load embedding model
    model_kwargs = {
        "model": embedding_model_path,
        "task": "embed",
        "device": 'cuda',
        "dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16,
        "max_model_len": MAX_LENGTH,
    }
    embedding_model = LLM(** model_kwargs)

    # Load RQVAE
    # Auto-detects config.json from parent directory
    config =  json.load(open(os.path.join(os.path.dirname(pretrained_rqvae_path) , "config.json"), "r", encoding="utf-8"))
    n_layers = config["n_layers"]
    codebook_size = config["codebook_size"]
    tokenizer = SemanticIdTokenizer(
        rqvae_weights_path=pretrained_rqvae_path,
        )
    tokenizer.rq_vae = accelerator.prepare(tokenizer.rq_vae)
    write_record_log("================" ,  log_path=record_path)
    write_record_log(f"Loaded {embedding_model_path} and {pretrained_rqvae_path} in {(time.time() - start_time):.2f} s" , log_path=record_path)

    total_sku_errors = 0
    json_files = glob.glob(str(Path(load_data_dir) / "*.json"))

    for input_file in json_files:
        file_start_time = time.time()
        file_name = os.path.basename(input_file)
        print(f"\nProcessing file: {file_name}")
        
        embedding_dataset = TemplatedTextDataset(
            file_path=input_file,
            text_template=TEXT_TEMPLATE,
            custom_fields=['sku']
        )
        template_error_count = embedding_dataset.template_error_count  
        total_lines = embedding_dataset.total_lines  
        
        num_workers = min(os.cpu_count() or 4, 16)
        embedding_dataloader = DataLoader(
            embedding_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=TemplatedTextDataset_fn,
            pin_memory=True,
            prefetch_factor=2
        )
        all_skus_file = []
        all_embeddings_file = []
        batch_times_file = []
        file_sku_errors = 0
        with torch.no_grad(): 
            for batch_idx, batch_dict in enumerate(tqdm(
                embedding_dataloader, total=len(embedding_dataloader), desc=f"Processing {file_name}"
            )):
                batch_start_time = time.time()
                
                # Extract text, sku, and metadata
                texts = batch_dict["text"]
                sku_strs = batch_dict["sku"]  
                # metadata = batch_dict["metadata"]  
                
                # Convert sku to int64, log errors
                batch_skus = []
                for sku_str in sku_strs:
                    try:
                        if sku_str is None:
                            raise ValueError("sku is empty")
                        sku_int = int(sku_str.strip())
                        batch_skus.append(sku_int)
                    except (ValueError, TypeError) as e:
                        file_sku_errors += 1
                        total_sku_errors += 1
                        print(f"Warning: sku conversion failed (value: {sku_str}, error: {e}), skipping")
                        batch_skus.append(-1)
                
                # Generate embeddings
                outputs = embedding_model.embed(texts)
                embeddings = torch.tensor([o.outputs.embedding for o in outputs])
                embeddings = embeddings.to(dtype=torch.float32) 
                
                # Filter out entries with sku conversion errors
                valid_mask = [sku != -1 for sku in batch_skus]
                valid_skus = [sku for sku, mask in zip(batch_skus, valid_mask) if mask]
                valid_embeddings = embeddings[valid_mask] 
                
                # Collect data
                all_skus_file.extend(valid_skus)
                all_embeddings_file.append(valid_embeddings)  
                
                batch_time = time.time() - batch_start_time
                batch_times_file.append(batch_time)
                
                del batch_dict, outputs, embeddings, valid_embeddings
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize() 
        
        if all_embeddings_file:
            file_embeddings = torch.cat(all_embeddings_file, dim=0).detach().cpu()  # shape：[N, embed_dim]
            file_skus = np.array(all_skus_file, dtype=np.int64)  # shape：[N]
            valid_count = len(file_skus)
        else:
            file_embeddings = torch.tensor([], dtype=torch.float32)
            file_skus = np.array([], dtype=torch.int64)
            valid_count = 0
        
        file_total_time = time.time() - file_start_time
        avg_batch_time = sum(batch_times_file) / len(batch_times_file) if batch_times_file else 0
        
        log_message = (f"File {file_name} embedding generation done - "
                      f"total time: {file_total_time:.4f}s, "
                      f"batch count: {len(embedding_dataloader)}, "
                      f"avg batch time: {avg_batch_time:.4f}s, "
                      f"template errors: {template_error_count}/{total_lines}, "
                      f"sku conversion errors: {file_sku_errors}/{valid_count + file_sku_errors}")
        print(log_message)
        write_record_log(log_message , log_path=record_path)
        # Build dataset from inference results using TigerDataset_Direct
        sid_start_time = time.time()
        raw_dataset = TigerDataset_Direct(embeddings= file_embeddings, skus=file_skus)
        sid_dataset = ItemData(raw_dataset, train_test_split="all")
        write_record_log(f"sid dataloader took {(time.time() - sid_start_time):.3f} s" , log_path = record_path)

        if accelerator.is_main_process:
            tokenizer.reset()
            tokenizer.rq_vae.eval()
            generate_start = time.time()
            corpus_ids , sku_ids = tokenizer.precompute_corpus_ids(sid_dataset)
            write_record_log(f"{input_file} precompute_corpus_ids took {(time.time() - generate_start):.3f}s")
            max_duplicates = corpus_ids[:, -1].max() / corpus_ids.shape[0]

            _, counts = torch.unique(
                corpus_ids[:, :], dim=0, return_counts=True
            )
            p = counts / corpus_ids.shape[0]
            rqvae_entropy = -(p * torch.log(p)).sum()

            for sid in range(n_layers):
                _, counts = torch.unique(corpus_ids[:, sid], return_counts=True)
                codebook_usage = len(counts) / codebook_size
                print(
                    f"Codebook Usage Layer {sid}: {codebook_usage:.4f} ({len(counts)}/{codebook_size})"
                )
                write_record_log(f'Codebook_Usage_Layer_{sid} : {codebook_usage:.4f}' , log_path=record_path)
            write_record_log(f"RQVAE Entropy : {(rqvae_entropy.cpu().item()):.6f}" , log_path=record_path)
            write_record_log(f"Max Duplicates : {(max_duplicates.cpu().item()):.6f}" , log_path=record_path)
            write_record_log(f"{file_name} Codebook computation/metrics total time {(time.time() - generate_start):.2f} s", log_path=record_path)
            
            if hasattr(corpus_ids, "cpu"):
                corpus_ids = corpus_ids.cpu().tolist()
            elif hasattr(corpus_ids, "numpy"):
                corpus_ids = corpus_ids.numpy().tolist()

            item_to_tokens , _ = process_rq_vae_codebook(corpus_ids , sku_ids)

            write_record_log(f"{file_name} Codebook generation total time {(time.time() - generate_start):.2f} s", log_path=record_path)
        
        # Save results as txt: sku, sid1, sid2, sid3, num_sid1, num_sid2, num_sid3
        save_path = os.path.join(save_data_dir, os.path.splitext(file_name)[0] + ".txt")
        with open(save_path, "w", encoding="utf-8") as f:
            for sku, tokens in item_to_tokens.items():
                # Parse tokens
                sid = "".join(tokens)
                sid1 , sid2 , sid3 = tokens
                num_sid1 , num_sid2 , num_sid3 = re.findall(r'<[a-zA-Z]+_(\d+)>', sid)
                # Write to file
                f.write(f"{sku}\t{sid1}\t{sid2}\t{sid3}\t{num_sid1}\t{num_sid2}\t{num_sid3}\n")
        
        write_record_log(f"{file_name} total time {(time.time() - file_start_time):.2f} s", log_path=record_path)




if __name__ == "__main__":
    config = parse_config()
    mode = config["mode"]
    if mode == "train":
        train_config = config["train"]
        train(**train_config)
    elif mode == "generate":
        generate_config = config["generate"]
        generate_emb(**generate_config)
    else:
        raise ValueError(f"Unknown mode: {mode}")
