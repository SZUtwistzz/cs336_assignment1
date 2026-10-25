# OpenWebText single-GPU training on enine

Repository: `/mnt/data/home/ysy0414/cs336_assignment1`

Token data: `/mnt/data/home/ysy0414/cs336_data/owt`

The OWT tokenizer has 32000 entries, with IDs 0 through 31999. Pass
`--vocab-size 32000` when starting a new run. Both `.bin` files contain
raw `uint16` token IDs. Training reads these files directly; it does not
retrain the tokenizer or tokenize the text again.

## Environment

From the repository directory, install the locked project environment if
it has not already been installed. This requires access to the package
indexes; uploading source code does not install dependencies.

```bash
cd /mnt/data/home/ysy0414/cs336_assignment1
python -m pip install uv
uv sync --locked
```

## Short validation run

Check that GPU 0 is available before starting. This is a separate 200-step
trial for checking CUDA, data, loss, memory use, and throughput.

```bash
cd /mnt/data/home/ysy0414/cs336_assignment1
CUDA_VISIBLE_DEVICES=0 uv run python -m cs336_basics.train_LM.training_together \
  --device cuda \
  --vocab-size 32000 \
  --train-path /mnt/data/home/ysy0414/cs336_data/owt/openwebtext_train_tokens.bin \
  --valid-path /mnt/data/home/ysy0414/cs336_data/owt/openwebtext_valid_tokens.bin \
  --run-dir "/mnt/data/home/ysy0414/cs336_assignment1/runs/owt_trial_$(date +%Y%m%d_%H%M%S)" \
  --batch-size 64 --num-steps 200 --warmup-steps 20 \
  --log-every 10 --eval-every 100 --eval-batches 5 --save-every 200
```

If this batch size does not fit, retry with a smaller batch size and a new
run directory. No throughput or GPU-memory claim has been measured on
this server yet. The script uses float32, as before.

The console prints training and validation losses. Each run's `metrics.csv`
records step and elapsed time; use stable intervals after startup to
estimate throughput. Evaluation and checkpointing also take time.

## Full run and resume

After the short run succeeds, start a NEW run with `--num-steps 20000`,
`--warmup-steps 200`, `--eval-every 200`, `--eval-batches 20`, and
`--save-every 1000`. At batch size 64 and context length 256, this
processes 327680000 tokens. These are starting settings, not tuned OWT
hyperparameters. A different batch size changes the token budget.

Do not extend the 200-step trial using `--resume`: its learning-rate
schedule was configured for only 200 steps. Resume an interrupted full
run from a checkpoint produced by that full run instead.

`--resume` restores the vocabulary size, training settings, model weights,
optimizer state, and step from the checkpoint and the adjacent run's
`config.json`. Keep the checkpoint and config together. Use a new output
directory when resuming. The current checkpoint format does not preserve
random-number state, so the sampled-batch stream is not exactly replayed.

TinyStories checkpoints using a 10000-token vocabulary are incompatible
with a 32000-token OWT run.

Default output is now `runs/train_<timestamp>` under the working directory.
Explicit data and output paths are expanded and saved as absolute paths.
The new vocabulary flag retains 10000 as its default for existing
TinyStories commands.
