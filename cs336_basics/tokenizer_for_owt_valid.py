from .tokenizer_experiments import tokenizer_experiment

opt_dataset_path ="E:/cs336/data/owt_valid.txt"
opt_vocab_path = "E:/cs336/saves/openwebtext_dataset/openwebtext_vocab.json"
opt_merges_path = "E:/cs336/saves/openwebtext_dataset/openwebtext_merges.txt"
opt_output_path = "E:/cs336/saves/token_array/openwebtext_valid_tokens.bin"
special_tokens = ["<|endoftext|>"]

tokenizer_experiment(opt_dataset_path,opt_vocab_path,opt_merges_path,opt_output_path,special_tokens)