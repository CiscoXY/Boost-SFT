# RQ-VAE for Semantic ID

## Requirements

```
pip install polars
pip install torch_geometric
```

## Usage

```bash
cd RQVAE
sh rqvae.sh
```

## Key Files

- `rqvae.py` — Entry point for training and SID generation
- `modules/` — Model components (encoder, quantizer, loss, transformer)
- `data/` — Dataset loading and preprocessing
