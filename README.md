# train


# inference

`generate_vllm` : 使用vllm进行generate，在空环境下(例如只有torch，而且版本和cuda也不对劲，简单来讲租个3090) , 直接
```python
pip install vllm
```
即可

# eval

`eval` : 具体进文件看

# model_tools

**在训练之前务必先进行add tokens和test tokens步骤**

# data

steps:

1. `sid_pt_to_csv` : 在rq-vae训练好后，将保存的.pt文件翻译成csv
2. `get_sid_understand_data` : 将拿到的csv和mata_data.json一起生成sid理解task的训练数据
3. `get_seq_data` : sft当中rec任务的数据
    3.1 `get_seq_data_dpo` : 适用于dpo的rec任务的数据(dpo阶段也应该只有rec任务)
4. `join_jsonl` 将2和3得到的数据join到一起，成为sft的训练数据