import math
import torch
import numpy as np
import os
import json

from data.schemas import SeqBatch
from data.schemas import TokenizedSeqBatch
from data.utils import item_batch_to , ItemBatch
from einops import rearrange
from einops import pack
from modules.utils import eval_mode
from modules.rqvae import RqVae
from typing import List
from typing import Optional
from torch import nn
from torch import Tensor
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler

BATCH_SIZE = 16


class SemanticIdTokenizer(nn.Module):
    """
    Tokenizes a batch of sequences of item features into a batch of sequences of semantic ids.

    Usage:
    1. Training mode: Pass all parameters to create a new model
    2. Inference mode: Only pass rqvae_weights_path to load a pretrained model
    """

    def __init__(
        self,
        rqvae_weights_path: Optional[str] = None,
        load_semantic_id_path: Optional[str] = None,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        hidden_dims: Optional[List[int]] = None,
        codebook_size: Optional[int] = None,
        n_layers: int = 3,
        n_cat_feats: int = 18,
        commitment_weight: float = 0.25,
        rqvae_codebook_normalize: bool = False,
        rqvae_sim_vq: bool = False,
    ) -> None:
        super().__init__()
        if rqvae_weights_path is not None:
            # Inference mode: Load from pretrained model
            self._load_from_pretrained(rqvae_weights_path)
        else:
            # Training mode: Create new model with provided parameters
            self._create_new_model(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
                codebook_size=codebook_size,
                n_layers=n_layers,
                n_cat_feats=n_cat_feats,
                commitment_weight=commitment_weight,
                rqvae_codebook_normalize=rqvae_codebook_normalize,
                rqvae_sim_vq=rqvae_sim_vq,
            )

        self.rq_vae.eval()
        self.reset()
        if load_semantic_id_path is not None:
            self.load_semantic_id(load_semantic_id_path)

    def load_semantic_id(self, load_semantic_id_path: str) -> None:
        tokenized_data = torch.load(load_semantic_id_path)
        self.cached_ids = torch.tensor(tokenized_data["semantic_ids"])

    def _load_from_pretrained(self, rqvae_weights_path: str) -> None:
        """Load model from pretrained weights and config."""
        rqvae_weights_dir = os.path.dirname(rqvae_weights_path)
        config_path = os.path.join(rqvae_weights_dir, "config.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path) as f:
            config = json.load(f)
        self.rq_vae = RqVae(**config)
        self.rq_vae.load_pretrained(rqvae_weights_path)

        # Set attributes from loaded config
        self.codebook_size = config.get("codebook_size")
        self.n_layers = config.get("n_layers", 3)

    def _create_new_model(
        self,
        input_dim: Optional[int],
        output_dim: Optional[int],
        hidden_dims: Optional[List[int]],
        codebook_size: Optional[int],
        n_layers: int,
        n_cat_feats: int,
        commitment_weight: float,
        rqvae_codebook_normalize: bool,
        rqvae_sim_vq: bool,
    ) -> None:
        """Create new model with provided parameters."""
        # Validate required parameters for training mode
        required_params = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dims": hidden_dims,
            "codebook_size": codebook_size,
        }

        missing_params = [
            name for name, value in required_params.items() if value is None
        ]
        if missing_params:
            raise ValueError(
                f"Training mode requires the following parameters: {', '.join(missing_params)}"
            )

        self.rq_vae = RqVae(
            input_dim=input_dim,
            embed_dim=output_dim,
            hidden_dims=hidden_dims,
            codebook_size=codebook_size,
            codebook_kmeans_init=False,
            codebook_normalize=rqvae_codebook_normalize,
            codebook_sim_vq=rqvae_sim_vq,
            n_layers=n_layers,
            n_cat_features=n_cat_feats,
            commitment_weight=commitment_weight,
        )

        # Set attributes
        self.codebook_size = codebook_size
        self.n_layers = n_layers

    def _get_hits(self, query: Tensor, key: Tensor) -> Tensor:
        return (rearrange(key, "b d -> 1 b d") == rearrange(query, "b d -> b 1 d")).all(
            axis=-1
        )

    def reset(self):
        self.cached_ids = None

    @property
    def sem_ids_dim(self):
        return self.n_layers
    """
    @torch.no_grad
    @eval_mode
    def precompute_corpus_ids(self, item_dataset) -> Tensor:
        cached_ids = None
        dedup_dim = []
        sku_list = [] # 存储sku
        sampler = BatchSampler(
            SequentialSampler(range(len(item_dataset))),
            batch_size=512,
            drop_last=False,
        )
        dataloader = DataLoader(
            item_dataset,
            sampler=sampler,
            shuffle=False,
            collate_fn=lambda x: x,
        )
        for batch in dataloader:
            batch_ids_tensor = torch.cat([item.ids for item in batch], dim=0)
            batch_x_tensor = torch.cat([item.x for item in batch], dim=0)
            batch_skus = []
            for item in batch:
                sku = item.sku_id
                if isinstance(sku, np.ndarray):
                    batch_skus.extend(sku.tolist())
                else:
                    batch_skus.append(sku)
            
            sku_list.extend(batch_skus)
            
            merged_batch = ItemBatch(
                ids=batch_ids_tensor,
                x=batch_x_tensor,
                sku_id=batch_skus 
            )
            
            batch_ids = self.forward(item_batch_to(merged_batch, self.rq_vae.device)).sem_ids
            # Detect in-batch duplicates
            is_hit = self._get_hits(batch_ids, batch_ids)
            hits = torch.tril(is_hit, diagonal=-1).sum(axis=-1)
            assert hits.min() >= 0
            if cached_ids is None:
                cached_ids = batch_ids.clone()
            else:
                # Detect batch-cache duplicates
                is_hit = self._get_hits(batch_ids, cached_ids)
                hits += is_hit.sum(axis=-1)
                cached_ids = pack([cached_ids, batch_ids], "* d")[0]
            dedup_dim.append(hits)
        # Concatenate new column to deduplicate ids
        dedup_dim_tensor = pack(dedup_dim, "*")[0]
        self.cached_ids = pack([cached_ids, dedup_dim_tensor], "b *")[0]

        return self.cached_ids , sku_list
    """
    @torch.no_grad
    @eval_mode
    def precompute_corpus_ids(self, item_dataset) -> Tensor:
        cached_ids = None
        sku_list = []  # 存储sku
        sampler = BatchSampler(
            SequentialSampler(range(len(item_dataset))),
            batch_size=512,
            drop_last=False,
        )
        dataloader = DataLoader(
            item_dataset,
            sampler=sampler,
            shuffle=False,
            collate_fn=lambda x: x,
        )
        for batch in dataloader:
            batch_ids_tensor = torch.cat([item.ids for item in batch], dim=0)
            batch_x_tensor = torch.cat([item.x for item in batch], dim=0)
            batch_skus = []
            for item in batch:
                sku = item.sku_id
                if isinstance(sku, np.ndarray):
                    batch_skus.extend(sku.tolist())
                else:
                    batch_skus.append(sku)
            
            sku_list.extend(batch_skus)
            
            merged_batch = ItemBatch(
                ids=batch_ids_tensor,
                x=batch_x_tensor,
                sku_id=batch_skus 
            )
            
            batch_ids = self.forward(item_batch_to(merged_batch, self.rq_vae.device)).sem_ids
            
            if cached_ids is None:
                cached_ids = batch_ids.clone()
            else:
                # 直接拼接，不检测重复
                cached_ids = pack([cached_ids, batch_ids], "* d")[0]
        
        self.cached_ids = cached_ids

        return self.cached_ids , sku_list

    @torch.no_grad
    @eval_mode
    def exists_prefix(self, sem_id_prefix: Tensor) -> Tensor:
        if self.cached_ids is None:
            raise Exception("No match can be found in empty cache.")

        prefix_length = sem_id_prefix.shape[-1]
        prefix_cache = self.cached_ids[:, :prefix_length]
        out = torch.zeros(
            *sem_id_prefix.shape[:-1], dtype=bool, device=sem_id_prefix.device
        )

        # Batch prefixes matching to avoid OOM.
        batches = math.ceil(sem_id_prefix.shape[0] // BATCH_SIZE)
        for i in range(batches):
            prefixes = sem_id_prefix[i * BATCH_SIZE : (i + 1) * BATCH_SIZE, ...]
            matches = (
                (prefixes.unsqueeze(-2) == prefix_cache.unsqueeze(-3))
                .all(axis=-1)
                .any(axis=-1)
            )
            out[i * BATCH_SIZE : (i + 1) * BATCH_SIZE, ...] = matches

        return out

    def _tokenize_seq_batch_from_cached(self, ids: Tensor) -> Tensor:
        return rearrange(
            self.cached_ids[ids.flatten(), :], "(b n) d -> b (n d)", n=ids.shape[1]
        )

    @torch.no_grad
    @eval_mode
    def forward(self, batch: SeqBatch) -> TokenizedSeqBatch:
        # TODO: Handle output inconstency in If-else.
        # If block has to return 3-sized ids for use in precompute_corpus_ids
        # Else block has to return deduped 4-sized ids for use in decoder training.
        sem_ids_fut = None
        if self.cached_ids is not None and self.cached_ids.device != batch.ids.device:
            self.cached_ids = self.cached_ids.to(batch.ids.device)
        if self.cached_ids is None or batch.ids.max() >= self.cached_ids.shape[0]:
            B, N = batch.ids.shape
            if hasattr(self.rq_vae, "get_semantic_ids"):
                sem_ids = self.rq_vae.get_semantic_ids(batch.x).sem_ids
            else:
                sem_ids = self.rq_vae.module.get_semantic_ids(batch.x).sem_ids
            D = sem_ids.shape[-1]
            seq_mask = None
        else:
            B, N = batch.ids.shape
            _, D = self.cached_ids.shape
            sem_ids = self._tokenize_seq_batch_from_cached(batch.ids)
            seq_mask = batch.seq_mask.repeat_interleave(D, dim=1)
            sem_ids[~seq_mask] = -1
            if batch.ids_fut is not None:
                sem_ids_fut = self._tokenize_seq_batch_from_cached(batch.ids_fut)

        token_type_ids = torch.arange(D, device=sem_ids.device).repeat(B, N)
        token_type_ids_fut = torch.arange(D, device=sem_ids.device).repeat(B, 1)

        return TokenizedSeqBatch(
            user_ids=batch.user_ids if hasattr(batch, "user_ids") else None,
            sem_ids=sem_ids,
            sem_ids_fut=sem_ids_fut,
            seq_mask=seq_mask,
            token_type_ids=token_type_ids,
            token_type_ids_fut=token_type_ids_fut,
        )


# if __name__ == "__main__":
#     dataset = ItemData("dataset/ml-1m-movie")
#     tokenizer = SemanticIdTokenizer(18, 32, [32], 32)
#     tokenizer.precompute_corpus_ids(dataset)

#     seq_data = SeqData("dataset/ml-1m")
#     batch = seq_data[:10]
#     tokenized = tokenizer(batch)
#     import pdb


#     pdb.set_trace()
