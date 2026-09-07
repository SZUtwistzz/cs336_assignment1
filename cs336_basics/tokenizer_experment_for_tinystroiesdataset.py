import random

from .tokenizer import Tokenizer

dataset_path ="E:/cs336/data/TinyStoriesV2-GPT4-valid.txt"
vocab_path = "E:/cs336/saves/tinystories_dataset/tinystories_vocab.json"
merges_path = "E:/cs336/saves/tinystories_dataset/tinystories_merges.txt"
special_tokens = ["<|endoftext|>"]

tokenizer_for_tinystories = Tokenizer.from_files(vocab_path,merges_path,special_tokens)

with open(dataset_path,'r',encoding="utf-8") as f:
        text = f.read()

docs = text.split("<|endoftext|>")

for doc in docs:
    print(doc)
    Encoded_tokens = tokenizer_for_tinystories.encode(doc)
    print(Encoded_tokens)
    print()

total_bytes = sum(len(doc.encode("utf-8")) for doc in docs)
total_tokens = sum(len(tokenizer_for_tinystories.encode(doc)) for doc in docs)
compresion_ratio = total_bytes/total_tokens
print(f"Cmpression Ratio :{compresion_ratio}")
    