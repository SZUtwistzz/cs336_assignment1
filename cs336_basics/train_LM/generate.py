import json
from pathlib import Path

import torch
from cs336_basics.transformer_LM.transformer_lm import transformer_LM
from cs336_basics.train_LM.decoding import decoding
from cs336_basics.tokenizer import Tokenizer

tns_vocab_path = "E:/cs336/saves/tinystories_dataset/tinystories_vocab.json"
tns_merges_path = "E:/cs336/saves/tinystories_dataset/tinystories_merges.txt"
special_tokens = ["<|endoftext|>"]

# 替换为你实际保存好的检查点路径
checkpoint_path = Path(
    "E:/cs336/saves/runs/tinystories_20260918_080507_208128/"
    "checkpoints/step_20000.pt"
)

config_path = checkpoint_path.parent.parent / "config.json"
with open(config_path, encoding="utf-8") as f:
    config = json.load(f)

# 先创建相同结构的模型
model = transformer_LM(
    vocab_size=config["vocab_size"],
    context_length=config["context_length"],
    num_layers=config["num_layers"],
    d_model=config["d_model"],
    num_heads=config["num_heads"],
    d_ff=config["d_ff"],
    theta=config["theta"],
)

# 再恢复权重
checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)
model.load_state_dict(checkpoint["model"])

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print(f"已加载第 {checkpoint['iteration']} 步的模型，设备：{device}")

prompt = str(input("prompt:"))
tokenizer = Tokenizer.from_files(vocab_filepath=tns_vocab_path,merges_filepath=tns_merges_path,special_tokens=special_tokens)
text = decoding(prompt=prompt,model=model,tokenizer=tokenizer,temperature=0.8,top_p=0.9,max_new_tokens=512,device=device)
print(f"result:{text}")
