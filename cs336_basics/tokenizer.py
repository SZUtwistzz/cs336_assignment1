from collections import Counter
from itertools import pairwise
import json
from typing import Iterable, Iterator
from .gpt2_bytes_to_unicode import gpt2_bytes_to_unicode
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    
    def __init__(self,vocab,merges,special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = list(special_tokens or [])

        self.token_to_id = {
            token_bytes: token_id
            for token_id,token_bytes in vocab.items()
        }

        next_id = max(self.vocab,default=-1)+1

        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")

            if token_bytes not in self.token_to_id:
                self.token_to_id[token_bytes] = next_id
                self.vocab[next_id] = token_bytes
                next_id+=1

        self.merges_ranks = {
            pair:rank
            for rank,pair in enumerate(merges)
        }

        self.special_token_to_id = {
            token:self.token_to_id[token.encode("utf-8")]
            for token in self.special_tokens
        }

    @classmethod
    def from_files(cls,vocab_filepath,merges_filepath,special_tokens=None):
        with open(vocab_filepath,"r",encoding="utf-8") as f:
            raw_vocab = json.load(f)

        byte_encoder = gpt2_bytes_to_unicode()
        byte_decoder = {
            unicode_character:byte_value
            for byte_value, unicode_character in byte_encoder.items()
        }

        vocab = {
            token_id:bytes(byte_decoder[character] for character in token_string)
            for token_string, token_id in raw_vocab.items()
        }

        merges = []
        with open(merges_filepath,"r",encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                left_string,right_string = line.split()
                left_bytes = bytes(
                    byte_decoder[character]
                    for character in left_string
                )
                right_bytes = bytes(
                    byte_decoder[character]
                    for character in right_string
                )

                merges.append((left_bytes,right_bytes))

        return cls(vocab,merges,special_tokens)
    
    def encode(self,text:str) ->list[int]:
        Encoded_tokens = []
        escaped_tokens = []

        sorted_special_tokens = sorted(
            self.special_tokens,
            key = len,
            reverse=True,
        )

        escaped_tokens = [
            re.escape(token)
            for token in sorted_special_tokens
        ]

        pattern = (
            "(" + "|".join(escaped_tokens) + ")"
            if escaped_tokens
            else None
        )

        text_pieces = re.split(pattern,text) if pattern else [text]

        for text_piece in text_pieces:
            if text_piece in self.special_token_to_id:
                Encoded_tokens.append(
                    self.special_token_to_id[text_piece]
                )
                continue

            words = re.finditer(PAT,text_piece)
            for word in words:
                pretoken_string = word.group(0)
                pretoken_bytes = pretoken_string.encode("utf-8")

                tokens = [
                    bytes([byte_value])
                    for byte_value in pretoken_bytes
                ]

                tokens = self._apply_bpe(tokens)
                Encoded_tokens.extend(
                self.token_to_id[token]
                for token in tokens
                )
        return Encoded_tokens
    
    def encode_iterable(self,iterable:Iterable[str]) ->Iterator[int]:
            for text in iterable:
                for token_id in self.encode(text):
                    yield token_id


    def decode(self,ids: list[int]) ->str:
        joined_bytes = b""
        list_of_bytes = []
        for token_id in ids:
            list_of_bytes.append(self.vocab[token_id])
        To_be_decoded = joined_bytes.join(list_of_bytes)
        return To_be_decoded.decode("utf-8",errors="replace")

    def _apply_bpe(self,tokens):
        while len(tokens)>=2:
            current_pairs = set(pairwise(tokens))

            #找出能够merge的pair
            candidate_pairs = [
                pair
                for pair in current_pairs
                if pair in self.merges_ranks
            ]

            #如果不存在可以merge的pair，退出循环
            if not candidate_pairs:
                break
    
            best_pair = min(
                candidate_pairs,
                key = lambda pair:self.merges_ranks[pair]
            )
    
            new_tokens = []
            position = 0
            merged_token = best_pair[0] + best_pair[1]
    
            while position < len(tokens):
                if (
                    position+1 < len(tokens)
                    and (tokens[position],tokens[position+1]) == best_pair 
                ):
                    new_tokens.append(merged_token)
                    position+=2
                else:
                    new_tokens.append(tokens[position])
                    position+=1
    
                tokens = new_tokens
        return tokens
