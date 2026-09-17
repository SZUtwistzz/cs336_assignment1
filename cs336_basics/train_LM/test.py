from cs336_basics.tokenizer import Tokenizer

opt_dataset_path ="E:/cs336/data/owt_train.txt"
opt_vocab_path = "E:/cs336/saves/openwebtext_dataset/openwebtext_vocab.json"
opt_merges_path = "E:/cs336/saves/openwebtext_dataset/openwebtext_merges.txt"
opt_output_path = "E:/cs336/saves/token_array/openwebtext_train_tokens.bin"
special_tokens = ["<|endoftext|>"]

mytokenizer = Tokenizer.from_files(opt_vocab_path,opt_merges_path,special_tokens)
list = mytokenizer.encode("<|endoftext|>")
print(len(list))