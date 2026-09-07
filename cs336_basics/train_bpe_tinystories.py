import json
from pathlib import Path

from .train_bpe import train_bpe
import time
import os
import psutil
from .gpt2_bytes_to_unicode import gpt2_bytes_to_unicode

input_path = "E:/cs336/data/TinyStoriesV2-GPT4-train.txt"
special_tokens = ["<|endoftext|>"]
vocab_size = 10000

def get_momery_GB():
    process = psutil.Process(os.getppid())
    return process.memory_info().rss/1024**3

def log_time(name,start):
    elapsed = time.perf_counter()-start
    print(f"[TIME] {name}: {elapsed:.2f}s")
    return time.perf_counter()

def token_bytes_to_printable(
    token: bytes,
    byte_encoder: dict[int, str],
) -> str:
    """
    将任意 bytes token 转换为 GPT-2 printable-unicode token。

    注意：
    这里绝对不能用 token.decode("utf-8")，
    因为 byte-level BPE token 不一定是合法 UTF-8。
    """
    return "".join(
        byte_encoder[b]
        for b in token
    )


def save_vocab(
    vocab: dict[int, bytes],
    output_path: str | Path,
) -> None:
    """
    保存为 Tokenizer.from_files 可以读取的 vocab JSON。

    训练时：
        token_id -> bytes

    文件中：
        printable token string -> token_id
    """
    output_path = Path(output_path)

    byte_encoder = gpt2_bytes_to_unicode()

    serializable_vocab = {}

    for token_id, token_bytes in vocab.items():
        token_string = token_bytes_to_printable(
            token_bytes,
            byte_encoder,
        )

        serializable_vocab[token_string] = token_id

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            serializable_vocab,
            f,
            ensure_ascii=False,
            indent=2,
        )


def save_merges(
    merges: list[tuple[bytes, bytes]],
    output_path: str | Path,
) -> None:
    """
    保存为 Tokenizer.from_files 可以读取的 merges TXT。

    每行格式：
        left_token right_token

    merge 的顺序必须严格保持不变，
    因为文件中的顺序就是 merge rank。
    """
    output_path = Path(output_path)

    byte_encoder = gpt2_bytes_to_unicode()

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        for left_bytes, right_bytes in merges:

            left = token_bytes_to_printable(
                left_bytes,
                byte_encoder,
            )

            right = token_bytes_to_printable(
                right_bytes,
                byte_encoder,
            )

            f.write(
                f"{left} {right}\n"
            )

start_total = time.perf_counter()
t=time.perf_counter()
print(f"Memory before training: {get_momery_GB()} GB")

vocab,merges = train_bpe(input_path,vocab_size,special_tokens)

t=log_time("Training",t)

print(f"Memory after training: {get_momery_GB()} GB")

longest_token_id, longest_token = max(
    vocab.items(),
    key=lambda item: len(item[1])
)

print("Longest token ID:", longest_token_id)
print("Length:", len(longest_token), "bytes")
print("Bytes:", longest_token)

save_vocab(vocab,"E:/cs336/saves/tinystories_dataset/tinystories_vocab.json")
save_merges(merges,"E:/cs336/saves/tinystories_dataset/tinystories_merges.txt")
