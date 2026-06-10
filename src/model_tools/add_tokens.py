from transformers import AutoTokenizer, AutoModelForCausalLM
import shutil

ORIGIN_MODEL = '/path/to/model'
NEW_MODEL = '/path/to/model-addtoken'

if shutil.os.path.exists(NEW_MODEL):
    shutil.rmtree(NEW_MODEL)
    print(f"Deleted old directory: {NEW_MODEL}")

tokenizer = AutoTokenizer.from_pretrained(
    ORIGIN_MODEL,
    legacy=True,
    use_fast=False,
    fix_mistral_regex=True
)
original_vocab_size = len(tokenizer)
print(f"Original vocab size: {original_vocab_size}")

token1 = [f'<a_{i}>' for i in range(256)]
token2 = [f'<b_{i}>' for i in range(256)]
token3 = [f'<c_{i}>' for i in range(256)]
new_tokens = token1 + token2 + token3
expected_added = len(new_tokens)
print(f"Expected number of tokens to add: {expected_added}")

num_added = tokenizer.add_tokens(new_tokens)
print(f"Actual number of tokens added: {num_added}")

assert num_added == expected_added, f"Failed to add tokens! Expected {expected_added}, actually added {num_added}"

new_vocab_size = len(tokenizer)
assert new_vocab_size == original_vocab_size + expected_added, \
    f"Vocab size mismatch! Expected {original_vocab_size + expected_added}, actual {new_vocab_size}"
print(f"Vocab size after adding: {new_vocab_size}")

model = AutoModelForCausalLM.from_pretrained(
    ORIGIN_MODEL,
    trust_remote_code=True
)

model.resize_token_embeddings(new_vocab_size)

embedding_dim = model.get_input_embeddings().weight.shape[0]
assert embedding_dim == new_vocab_size, \
    f"Model embedding dimension does not match tokenizer! Embedding dim {embedding_dim}, tokenizer vocab {new_vocab_size}"
print(f"Model embedding dimension: {embedding_dim} (aligned with tokenizer)")

sample_token = '<a_0><b_0><c_0>,<a_255><b_255><c_255>,<a_102><b_39><c_123>'
test_text = f"{sample_token} This is a test text"
tokens = tokenizer.tokenize(test_text)
print(f"\nTest text tokenization result (first 10 tokens): {tokens[:10]}")
for t in ['<a_0>', '<b_0>', '<c_0>']:
    assert t in tokens, f"New token {t} was not correctly recognized! Not found in tokenization result"
print("✅ New token recognition OK")

tokenizer.save_pretrained(NEW_MODEL)
model.save_pretrained(NEW_MODEL)
print(f"\n✅ Aligned model and tokenizer saved to: {NEW_MODEL}")
print(f"Final confirmation: Tokenizer vocab={len(tokenizer)}, model embedding={model.get_input_embeddings().weight.shape[0]}")