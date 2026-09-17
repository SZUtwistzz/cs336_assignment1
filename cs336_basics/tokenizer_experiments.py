import os
from .tokenizer import Tokenizer
import numpy as np

def tokenizer_experiment(
            dataset_path:str,
            vocab_path:str,
            merges_path:str,
            output_path:str,
            special_tokens,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    
    tokenizer = Tokenizer.from_files(vocab_path,merges_path,special_tokens)
    assert max(tokenizer.vocab) <= np.iinfo(np.uint16).max
    
    total_tokens = 0
    buffer = []
    buffer_size = 1_000_000
    print("Running tokenizer...")
    with open(dataset_path,'r',encoding="utf-8") as input_file,\
        open(output_path,"wb") as output_file:

        for token_id in tokenizer.encode_iterable(input_file):
            buffer.append(token_id)
            total_tokens+=1

            if len(buffer) >=buffer_size:
                np.asarray(buffer,dtype=np.uint16).tofile(output_file)
                buffer.clear()

        if buffer:
            np.asarray(buffer,dtype=np.uint16).tofile(output_file)

    total_bytes = os.path.getsize(dataset_path)
            
    compresion_ratio = total_bytes/total_tokens
    print(f"Cmpression Ratio :{compresion_ratio}")
      
tns_dataset_path ="E:/cs336/data/TinyStoriesV2-GPT4-train.txt"
tns_vocab_path = "E:/cs336/saves/tinystories_dataset/tinystories_vocab.json"
tns_merges_path = "E:/cs336/saves/tinystories_dataset/tinystories_merges.txt"
tns_output_path = "E:/cs336/saves/token_array/tinystories_train_tokens.bin"
special_tokens = ["<|endoftext|>"]

opt_dataset_path ="E:/cs336/data/owt_train.txt"
opt_vocab_path = "E:/cs336/saves/openwebtext_dataset/openwebtext_vocab.json"
opt_merges_path = "E:/cs336/saves/openwebtext_dataset/openwebtext_merges.txt"
opt_output_path = "E:/cs336/saves/token_array/openwebtext_train_tokens.bin"
# tokenizer_experiment(tns_dataset_path,tns_vocab_path,tns_merges_path,tns_output_path,special_tokens)


    