# requairements

```
pip install polars
pip install torch_geometric
```

# 启动方式

```python
cd RQVAE
sh rqvae.sh
```

# 重要文件:

* `./config`: 存储配置文件
    * `rqvae.json` : 分为3种模式 
        1. `train` -- 训练，数据为pt格式，具体一点每个pt是个dict，包含两个字段 `sku` 和 `embedding` , shape为`N`和(`N`,`embedding_dim`)
        2. `generate` -- `sid`生成(推理), 输入的数据和训练的结构一致
* `./rqvae.py` : 入口文件，要改的话基本就是改这个