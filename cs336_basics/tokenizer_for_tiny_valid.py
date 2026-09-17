from .tokenizer_experiments import tokenizer_experiment

tns_dataset_path ="E:/cs336/data/TinyStoriesV2-GPT4-valid.txt"
tns_vocab_path = "E:/cs336/saves/tinystories_dataset/tinystories_vocab.json"
tns_merges_path = "E:/cs336/saves/tinystories_dataset/tinystories_merges.txt"
tns_output_path = "E:/cs336/saves/token_array/tinystories_valid_tokens.bin"
special_tokens = ["<|endoftext|>"]

tokenizer_experiment(tns_dataset_path,tns_vocab_path,tns_merges_path,tns_output_path,special_tokens)