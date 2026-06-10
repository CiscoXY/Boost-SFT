from transformers import AutoTokenizer

ORIGIN_MODEL = "/path/to/models/checkpoint-XXXXX"

tokenizer = AutoTokenizer.from_pretrained(ORIGIN_MODEL, legacy=True, use_fast=False)
print('Vocab size:', len(tokenizer))

sample_token = '<a_0><b_0><c_0>,<a_255><b_255><c_255>,<a_102><b_39><c_123>'
test_text = f"{sample_token} This is a test text (1)"
print(f"Test text {sample_token}")
tokens = tokenizer.tokenize(test_text)
print(f"Tokenization result: {tokens}")


sample_token = '<a_682><b_266><c_496>,<a_410><b_171><c_944>,<a_682><b_421><c_69>,<a_160><b_546><c_949>,<a_381><b_465><c_177>,<a_750><b_91><c_427>,<a_750><b_91><c_427>,<a_750><b_91><c_427>,<a_750><b_91><c_427>,<a_750><b_91><c_427>,<a_750><b_91><c_427>,<a_750><b_91><c_427>,<a_750><b_91><c_427>,<a_750><b_500><c_453>,'
test_text = f"{sample_token} This is a test text (2)"
print(f"Test text {sample_token}")
tokens = tokenizer.tokenize(test_text)
print(f"Tokenization result: {tokens}")