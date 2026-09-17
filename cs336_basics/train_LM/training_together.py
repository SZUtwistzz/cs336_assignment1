from .data_loader import data_loader
from cs336_basics.transformer_LM.transformer_lm import transformer_LM 
from .AdamW import adamw
from .checkpointing import save_checkpoint,load_checkpoint
from .gradient_clipping import gradient_clipping
from .learning_rate_schedule import learning_rate_schedule
from cs336_basics.train_LM.cross_entropy import cross_entropy
import numpy as np
import torch
import argparse
import time
import pathlib
import csv
import json

PathOfCheckingPoint = "E:/cs336/assignment1-basics/cs336_basics/checkingpoint/"
DataPath = "E:/cs336/saves/token_array/tinystories_train_tokens.bin"
validPath = "E:/cs336/saves/token_array/tinystories_valid_tokens.bin"

from pathlib import Path
from datetime import datetime

parser = argparse.ArgumentParser(description="TinyStories training")
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--num-steps", type=int, default=200)
parser.add_argument("--max-lr", type=float, default=3e-4)
parser.add_argument("--min-lr", type=float, default=3e-5)
parser.add_argument("--warmup-steps", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--train-path", type=str, default=DataPath)
parser.add_argument("--valid-path", type=str, default=validPath)
parser.add_argument("--log-every", type=int, default=10)
parser.add_argument("--eval-every", type=int, default=50)
parser.add_argument("--eval-batches", type=int, default=5)
parser.add_argument("--save-every", type=int, default=100)
parser.add_argument("--run-dir", type=str, default=None)
parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
args = parser.parse_args()

if args.device == "cuda" and not torch.cuda.is_available():
    parser.error("CUDA is unavailable. Install a CUDA-enabled PyTorch build and check the NVIDIA driver.")

if args.batch_size <= 0:
    parser.error("--batch-size 必须大于 0")
if not 0 <= args.warmup_steps < args.num_steps:
    parser.error("需要满足 0 <= warmup-steps < num-steps")
if not 0 <= args.min_lr <= args.max_lr:
    parser.error("需要满足 0 <= min-lr <= max-lr")
if min(
    args.log_every,
    args.eval_every,
    args.eval_batches,
    args.save_every,
) <= 0:
    parser.error("日志、验证、保存间隔及验证批次数必须大于 0")

# 每次实验单独保存，避免混合日志或覆盖旧检查点。
run_dir = (
    Path(args.run_dir)
    if args.run_dir
    else Path("E:/cs336/saves/runs")
    / datetime.now().strftime("tinystories_%Y%m%d_%H%M%S_%f")
)
run_dir.mkdir(parents=True, exist_ok=False)
checkpoint_dir = run_dir / "checkpoints"
checkpoint_dir.mkdir()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
print(f"Training device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
epochs = args.num_steps
batch_size = args.batch_size
max_lr = args.max_lr
min_lr = args.min_lr

vocab_size = 10000
context_length = 256
num_layers = 4
d_model = 512
num_heads = 16
d_ff = 1344
betas = (0.9, 0.999)
weight_decay = 0.01
eps = 1e-8
norm_M = 1.0
theta = 10000.0

data = np.memmap(args.train_path, mode="r", dtype=np.uint16)
data_valid = np.memmap(args.valid_path, mode="r", dtype=np.uint16)

if len(data) <= context_length or len(data_valid) <= context_length:
    raise ValueError("训练集和验证集都需要至少 context_length + 1 个 token")

config = vars(args).copy()
config.update({
    "run_dir": str(run_dir),
    "device": device,
    "dtype": "float32",
    "data_dtype": "uint16",
    "vocab_size": vocab_size,
    "context_length": context_length,
    "num_layers": num_layers,
    "d_model": d_model,
    "num_heads": num_heads,
    "d_ff": d_ff,
    "betas": betas,
    "weight_decay": weight_decay,
    "eps": eps,
    "gradient_clip_norm": norm_M,
    "theta": theta,
    "cosine_end_step": epochs,
})

with open(run_dir / "config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"本次实验保存位置：{run_dir}")



model = transformer_LM(vocab_size = vocab_size,context_length = context_length,num_layers = num_layers,d_model = d_model,num_heads = num_heads,d_ff = d_ff,theta = theta)
model.to(device)
optimizer = adamw(model.parameters(),lr=max_lr,betas=betas,weight_decay=weight_decay,eps = eps)

if device == "cuda":
    torch.cuda.synchronize()
start_time = time.perf_counter()
csv_path = run_dir / "metrics.csv"
f = open(csv_path,mode="w",encoding="utf-8",newline='')
writer = csv.DictWriter(f=f,fieldnames=["step","elapsed_seconds","split","loss","lr"])
writer.writeheader()



for epoch in range(epochs):
    completed_steps = epoch + 1

    model.train()

    inputs,lables = data_loader(data,batch_size,context_length,device)

    inputs,lables = inputs.to(device),lables.to(device)

    lr = learning_rate_schedule(epoch,max_lr,min_lr,Tw=args.warmup_steps,Tc=epochs)

    optimizer.zero_grad()
    for group in optimizer.param_groups:
        group["lr"] = lr
        
    outputs = model(inputs)

    loss = cross_entropy(outputs,lables)

    loss.backward()
    gradient_clipping(model.parameters(),norm_M)
    optimizer.step()

    if completed_steps % args.log_every == 0 or completed_steps == epochs:
        train_loss = loss.item()
        if device == "cuda":
            torch.cuda.synchronize()
        writer.writerow({"step":completed_steps,"elapsed_seconds":time.perf_counter()-start_time,"split":"train","loss":train_loss,"lr":lr})
        f.flush()
        print(f"step={completed_steps}/{epochs} train_loss={train_loss:.4f} lr={lr:.6g}")

    if completed_steps % args.eval_every == 0 or completed_steps == epochs:
        model.eval()
        running_loss = 0.0
        # Validation sampling must not change subsequent training batches.
        training_rng_state = np.random.get_state()
        for valid_epoch in range(args.eval_batches):
            with torch.no_grad():
                inputs,lables = data_loader(data_valid,batch_size,context_length,device)
                inputs,lables =  inputs.to(device), lables.to(device)

                outputs = model(inputs)
                loss = cross_entropy(outputs,lables)
            running_loss+=loss.item()
        np.random.set_state(training_rng_state)
        running_loss/=args.eval_batches
        print(f"valid loss:{running_loss}")
        model.train()

        if device == "cuda":
            torch.cuda.synchronize()
        writer.writerow({
            "step": completed_steps,
            "elapsed_seconds": time.perf_counter() - start_time,
            "split": "valid",
            "loss": running_loss,
            "lr": lr,
        })
        f.flush()

    if completed_steps % args.save_every == 0 or completed_steps == epochs:
        savapath = checkpoint_dir / f"step_{completed_steps}.pt"
        save_checkpoint(model=model,optimizer=optimizer,iteration=completed_steps,out=savapath)
        print(f"checkpoint: {savapath}")

f.close()

    
