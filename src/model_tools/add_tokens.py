from transformers import AutoTokenizer, AutoModelForCausalLM
import shutil

ORIGIN_MODEL = '/mnt/Qwen3-0.6B'
NEW_MODEL = '/mnt/Qwen3-0.6B-addtoken'

if shutil.os.path.exists(NEW_MODEL):
    shutil.rmtree(NEW_MODEL)
    print(f"已删除旧目录: {NEW_MODEL}")

# ---------------------- 加载原始Tokenizer（保持参数一致）----------------------
tokenizer = AutoTokenizer.from_pretrained(
    ORIGIN_MODEL,
    legacy=True,  
    use_fast=False,  
    fix_mistral_regex=True  # Qwen模型必需（匹配训练时的参数）
)
original_vocab_size = len(tokenizer)
print(f"原始Vocab大小: {original_vocab_size}")  # 记录原始大小（比如151643）

# ---------------------- 生成并添加新token ----------------------
token1 = [f'<a_{i}>' for i in range(256)] 
token2 = [f'<b_{i}>' for i in range(256)] 
token3 = [f'<c_{i}>' for i in range(256)] 
new_tokens = token1 + token2 + token3
expected_added = len(new_tokens)
print(f"预期添加token数: {expected_added}")

# 添加token（replace=False确保不重复添加，返回实际添加数量）
num_added = tokenizer.add_tokens(new_tokens)
print(f"实际添加token数: {num_added}")

assert num_added == expected_added, f"添加token失败！预期{expected_added}个，实际添加{num_added}个"

# 校验：添加后Vocab大小 = 原始大小 + 新增数量
new_vocab_size = len(tokenizer)
assert new_vocab_size == original_vocab_size + expected_added, \
    f"Vocab大小异常！预期{original_vocab_size + expected_added}，实际{new_vocab_size}"
print(f"添加后Vocab大小: {new_vocab_size}") 

# ---------------------- 加载模型并调整嵌入层 ----------------------
model = AutoModelForCausalLM.from_pretrained(
    ORIGIN_MODEL,
    trust_remote_code=True  # Qwen模型必需
)

# 关键：调整嵌入层大小，确保和Tokenizer vocab一致
model.resize_token_embeddings(new_vocab_size)

# 校验：模型嵌入层维度必须 == Tokenizer vocab大小
embedding_dim = model.get_input_embeddings().weight.shape[0]
assert embedding_dim == new_vocab_size, \
    f"模型嵌入层与Tokenizer不匹配！嵌入层维度{embedding_dim}，Tokenizer vocab{new_vocab_size}"
print(f"模型嵌入层维度: {embedding_dim}（已和Tokenizer对齐）")

# ---------------------- 测试 ----------------------
sample_token = '<a_0><b_0><c_0>,<a_255><b_255><c_255>,<a_102><b_39><c_123>'
test_text = f"{sample_token} 这是一段测试文本"
tokens = tokenizer.tokenize(test_text)
print(f"\n测试文本tokenization结果（前10个token）: {tokens[:10]}")
# 验证新增token是否被正确识别（不应被拆分）
for t in ['<a_0>', '<b_0>', '<c_0>']:
    assert t in tokens, f"新增token {t} 未被正确识别！tokenization结果中无此token"
print("✅ 新增token识别正常")

# ---------------------- 保存对齐后的模型和Tokenizer ----------------------
tokenizer.save_pretrained(NEW_MODEL)
model.save_pretrained(NEW_MODEL)
print(f"\n✅ 对齐后的模型和Tokenizer已保存到: {NEW_MODEL}")
print(f"最终确认：Tokenizer vocab={len(tokenizer)}，模型嵌入层={model.get_input_embeddings().weight.shape[0]}")