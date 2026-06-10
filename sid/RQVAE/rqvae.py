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
from modules.rqvae import RqVae                             # 必需品
from modules.quantize import QuantizeForwardMode            # 必需品
from modules.tokenizer.semids import SemanticIdTokenizer    # 必须品
from modules.utils import parse_config                       # 必需品
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
    print(f"创建rar_dataset耗时{(time.time() - raw_dataset_start):.2f} s")
    write_record_log(f"创建rar_dataset耗时{(time.time() - raw_dataset_start):.2f} s" , log_path=record_path)

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
    print(f"创建train_dataloader耗时{(time.time() - train_dataloader_start):.2f} s")
    write_record_log(f"创建train_dataloader耗时{(time.time() - train_dataloader_start):.2f} s" , log_path=record_path)


    eval_dataloader_start = time.time()

    eval_dataset = ItemData(raw_dataset, train_test_split="eval")
    eval_sampler = BatchSampler(RandomSampler(eval_dataset), batch_size, False)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=None,
        collate_fn=lambda batch: batch,
    )
    print(f"创建eval_dataloader耗时{(time.time() - eval_dataloader_start):.2f} s")
    write_record_log(f"创建eval_dataloader耗时{(time.time() - eval_dataloader_start):.2f} s" , log_path=record_path)

    train_loader_prepare = time.time()

    train_dataloader = accelerator.prepare(train_dataloader)
    print(f"创建train_dataloader_prepare耗时{(time.time() - train_loader_prepare):.2f} s")
    write_record_log(f"创建train_dataloader_prepare耗时{(time.time() - train_loader_prepare):.2f} s ", log_path=record_path)

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
                write_record_log(f"Eval Res - Iter {iter+1} 总耗时 {(eval_end - eval_start):.2f} s", log_path=record_path)

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
    这个generate是 embedding -> sid的generate

    load_data_dir : 保存有sku 、 embedding字段的.pt文件的文件夹
    k_files_per_save : 每k个pt文件打包成一个TigerDataset进行推理，减少内存占用和可能的异常
    save_data_dir : 保存结果的文件夹
    pretrained_rqvae_paths : 预训练好的rqvae模型路径
    record_path : 记录日志的文件
    embedding_size : 预先留存的巨大tensor，减少整合tensor的耗时
    """
    accelerator = Accelerator()
    device = accelerator.device
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)
    def process_single_model(model_path , full_dataset , save_sign):
        """
        获取model_path的父路径，然后读取父路径下的config.json
        然后根据config.json中的参数，初始化模型
        """
        config =  json.load(open(os.path.join(os.path.dirname(model_path) , "config.json"), "r", encoding="utf-8"))
        n_layers = config["n_layers"]
        codebook_size = config["codebook_size"]

        save_data_dir4model = os.path.join(save_data_dir, os.path.splitext(os.path.basename(model_path))[0])
        if not os.path.exists(save_data_dir4model):
            os.makedirs(save_data_dir4model)

        start_time = time.time()
        # 输入一个model_path就行，他会自动捕捉父路径下的config.json
        tokenizer = SemanticIdTokenizer(
            rqvae_weights_path=model_path,
            )
        tokenizer.rq_vae = accelerator.prepare(tokenizer.rq_vae)
        write_record_log("================" ,  log_path=record_path)
        write_record_log(f"加载 {model_path} 耗时{(time.time() - start_time):.2f} s" , log_path=record_path)

        if accelerator.is_main_process:
            tokenizer.reset()
            tokenizer.rq_vae.eval()

            write_record_log(f"{model_path} generate start ......" , log_path=record_path)
            generate_start = time.time()
            corpus_ids , sku_ids = tokenizer.precompute_corpus_ids(full_dataset)
            write_record_log(f"{model_path} precomput_corpus_ids 耗时 {(time.time() - generate_start):.3f}")
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
            write_record_log(f"{model_path} 码本计算/指标总耗时 {(time.time() - generate_start):.2f} s", log_path=record_path)
            
            if hasattr(corpus_ids, "cpu"):
                corpus_ids = corpus_ids.cpu().tolist()
            elif hasattr(corpus_ids, "numpy"):
                corpus_ids = corpus_ids.numpy().tolist()

            (
            item_to_tokens,
            tokens_to_item,
            ) = process_rq_vae_codebook(corpus_ids , sku_ids)
            write_record_log(f"{model_path} 码本生成总耗时 {(time.time() - generate_start):.2f} s", log_path=record_path)
            save_data_dict = {
                "item_to_tokens": item_to_tokens,
                "tokens_to_item": tokens_to_item,
            }
            torch.save(
                save_data_dict, os.path.join(save_data_dir4model, f"saved_tokenizer_data{save_sign}.pt")
            )
            write_record_log(f"{model_path} 总耗时 {(time.time() - generate_start):.2f} s", log_path=record_path)
    
    pt_files = glob.glob(str(Path(load_data_dir) / "*.pt"))
    # 每k个pt文件打包成一个TigerDataset进行推理，减少内存占用和可能的异常
    pt_file_batchs = [pt_files[i:i+k_files_per_save] for i in range(0, len(pt_files), k_files_per_save)]
    for idx , pt_file_batch in enumerate(pt_file_batchs):
        gc.collect()  # 清理Python内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 清理未使用的CUDA显存
            torch.cuda.ipc_collect()  # 清理进程间通信残留的显存
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
    这个generate是 sku info(json文件，包含sku和text template的必须字段) -> sid(txt)的 端到端sid生成 , 主要处理增量更新
    因此只支持一个embedding model 以及 一个pretraiedn_rqvae_path

    load_data_dir : 保存有sku信息的文件夹(只支持json格式)
    save_data_dir : 保存结果的文件夹，保存格式为txt，原始字段为7个，sku , sid1 , sid2 , sid3 , num_sid1 , num_sid2 , num_sid3，用斜杠t分隔
    embedding_model_path : 做embedding的模型
    pretrained_rqvae_paths : 预训练好的rqvae模型路径
    record_path : 记录日志的文件
    """
    accelerator = Accelerator()
    device = accelerator.device
    os.makedirs(save_data_dir, exist_ok=True)

    BATCH_SIZE = 10_0000
    TEXT_TEMPLATE = """
    商品名: {item_name}
    所属三级类目: {cid1_name} | {cid2_name} | {cid3_name}
    """.strip()
    MAX_LENGTH = 1024
    def TemplatedTextDataset_fn(batch):
        """批量处理函数：收集文本、sku和元数据"""
        texts = [item["text"] for item in batch]
        skus = [item["sku"] for item in batch]  # 单独提取sku
        # 保留其他元数据（如果后续需要）
        metadata = [
            {k: v for k, v in item.items() if k not in ["text", "sku"]} 
            for item in batch
        ]
        return {"text": texts, "sku": skus, "metadata": metadata}
    

    start_time = time.time()
    # 加载embedding model
    model_kwargs = {
        "model": embedding_model_path,
        "task": "embed",
        "device": 'cuda',
        "dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16,
        "max_model_len": MAX_LENGTH,
    }
    embedding_model = LLM(** model_kwargs)

    # 加载RQVAE
    # 输入一个model_path就行，他会自动捕捉父路径下的config.json
    config =  json.load(open(os.path.join(os.path.dirname(pretrained_rqvae_path) , "config.json"), "r", encoding="utf-8"))
    n_layers = config["n_layers"]
    codebook_size = config["codebook_size"]
    tokenizer = SemanticIdTokenizer(
        rqvae_weights_path=pretrained_rqvae_path,
        )
    tokenizer.rq_vae = accelerator.prepare(tokenizer.rq_vae)
    write_record_log("================" ,  log_path=record_path)
    write_record_log(f"加载 {embedding_model_path} 和 {pretrained_rqvae_path} 耗时{(time.time() - start_time):.2f} s" , log_path=record_path)

    total_sku_errors = 0
    json_files = glob.glob(str(Path(load_data_dir) / "*.json"))

    for input_file in json_files:
        file_start_time = time.time()
        file_name = os.path.basename(input_file)
        print(f"\n开始处理文件: {file_name}")
        
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
            pin_memory=True,  # 启用内存固定 , 默认使用GPU了
            prefetch_factor=2  # 预加载下一批数据
        )
        all_skus_file = []
        all_embeddings_file = []
        batch_times_file = []
        file_sku_errors = 0
        with torch.no_grad(): 
            for batch_idx, batch_dict in enumerate(tqdm(
                embedding_dataloader, total=len(embedding_dataloader), desc=f"处理 {file_name}"
            )):
                batch_start_time = time.time()
                
                # 提取文本、sku和元数据
                texts = batch_dict["text"]
                sku_strs = batch_dict["sku"]  
                # metadata = batch_dict["metadata"]  
                
                # sku：转换为int64，记录转换错误
                batch_skus = []
                for sku_str in sku_strs:
                    try:
                        if sku_str is None:
                            raise ValueError("sku为空")
                        sku_int = int(sku_str.strip())
                        batch_skus.append(sku_int)
                    except (ValueError, TypeError) as e:
                        file_sku_errors += 1
                        total_sku_errors += 1
                        print(f"警告：sku转换失败（值：{sku_str}，错误：{e}），跳过该条目")
                        # 用特殊值标记错误（后续过滤）
                        batch_skus.append(-1)
                
                # 生成embedding
                outputs = embedding_model.embed(texts)
                embeddings = torch.tensor([o.outputs.embedding for o in outputs])
                embeddings = embeddings.to(dtype=torch.float32) 
                
                # 过滤掉sku转换错误的条目（避免无效数据）
                valid_mask = [sku != -1 for sku in batch_skus]
                valid_skus = [sku for sku, mask in zip(batch_skus, valid_mask) if mask]
                valid_embeddings = embeddings[valid_mask] 
                
                # 收集数据
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
        
        log_message = (f"文件 {file_name} embedding生成完成 - "
                      f"总耗时: {file_total_time:.4f}s, "
                      f"batch数量: {len(embedding_dataloader)}, "
                      f"平均batch耗时: {avg_batch_time:.4f}s, "
                      f"模板应用错误行数: {template_error_count}/{total_lines}, "
                      f"sku转换错误数: {file_sku_errors}/{valid_count + file_sku_errors}")
        print(log_message)
        write_record_log(log_message , log_path=record_path)
        # 推理出来的结果用TigerDataset_Direct构成dataset，需要注意的是embedding的shape是 [len_dataloader,embedding_dim] , sku则是int64(得转换一下) , 然后是numpy的array
        sid_start_time = time.time()
        raw_dataset = TigerDataset_Direct(embeddings= file_embeddings, skus=file_skus)
        sid_dataset = ItemData(raw_dataset, train_test_split="all")
        write_record_log(f"sid dataloader耗时 : {(time.time() - sid_start_time):.3f} s" , log_path = record_path)

        if accelerator.is_main_process:
            tokenizer.reset()
            tokenizer.rq_vae.eval()
            generate_start = time.time()
            corpus_ids , sku_ids = tokenizer.precompute_corpus_ids(sid_dataset)
            write_record_log(f"{input_file} precomput_corpus_ids 耗时 {(time.time() - generate_start):.3f}")
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
            write_record_log(f"{file_name} 码本计算/指标总耗时 {(time.time() - generate_start):.2f} s", log_path=record_path)
            
            if hasattr(corpus_ids, "cpu"):
                corpus_ids = corpus_ids.cpu().tolist()
            elif hasattr(corpus_ids, "numpy"):
                corpus_ids = corpus_ids.numpy().tolist()

            item_to_tokens , _ = process_rq_vae_codebook(corpus_ids , sku_ids)

            write_record_log(f"{file_name} 码本生成总耗时 {(time.time() - generate_start):.2f} s", log_path=record_path)
        
        # 结果保存 也就是item_to_tokens这个dict的结果保存成txt 七个字段 sku , sid1 , sid2 , sid3 , num_sid1 , num_sid2 , num_sid3 ，key是sku，value是个str，得做解析
        save_path = os.path.join(save_data_dir, os.path.splitext(file_name)[0] + ".txt")
        with open(save_path, "w", encoding="utf-8") as f:
            for sku, tokens in item_to_tokens.items():
                # 解析tokens
                sid = "".join(tokens)
                sid1 , sid2 , sid3 = tokens
                num_sid1 , num_sid2 , num_sid3 = re.findall(r'<[a-zA-Z]+_(\d+)>', sid)
                # 写入文件
                f.write(f"{sku}\t{sid1}\t{sid2}\t{sid3}\t{num_sid1}\t{num_sid2}\t{num_sid3}\n")
        
        write_record_log(f"{file_name} 总耗时 {(time.time() - file_start_time):.2f} s", log_path=record_path)




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
