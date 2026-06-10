# train

在train中，只有sft_attenuation.py是可以直接脚本启动的，其他的具体配置都写死在代码里了

`sft_attenuation` : 3stage的衰减版本sft
`sft_llamafactory` : 使用llamafactory的最传统sft
`dpo` : lora版本
`dpo_full` : full版本
`sft_loss_version` : mask掉input loss，并且完全衰减到指定weights的版本

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