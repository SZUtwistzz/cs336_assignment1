from itertools import pairwise
from .pretokenization_example import find_chunk_boundaries

import regex as re
from collections import Counter, defaultdict

def train_bpe(
    input_path: str,
    vocab_size: int,   
    special_tokens: list[str]
) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    # 建立一个空字典 vocab。
    # 遍历整数 0～255。
    # 每个整数都要转换成“长度为 1 的 bytes 对象”。- 应使用“包含该整数的单元素序列”构造 bytes。
    # - 不要直接把整数交给 bytes，否则整数会被解释为需要创建多少个零字节。

    # 将 byte 的整数值作为 key，单字节 bytes 作为 value。
    # 此时检查 len(vocab) == 256。
    # 遍历 special_tokens：- 使用 UTF-8 将字符串整体编码为 bytes；
    # - 使用当前 vocabulary 的下一个可用整数作为 ID；
    # - 不要把 special token 拆成单字节。

    # 初始化 merges 时，它应是一个空列表。
    Vocabulary = {i:bytes([i]) for i in range(0,256)}
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    if len(Vocabulary)!=256:
        print("Fail to init")
        return
    
    current_index = 256
    for special_token in special_tokens:
        Vocabulary[current_index] = special_token.encode("utf-8")
        current_index+=1

    merges = []

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        frequency_table = Counter()
        escaped_tokens = []

        #遍历所有special token 并得到pattern
        for special_token in special_tokens:
            escaped_tokens.append(re.escape(special_token))

        pattern = "|".join(escaped_tokens)
        
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            ##pattern = "|".join(special_tokens) 只有一个special_token不需要join
            text_pieces = re.split(pattern,chunk) if pattern else [chunk]

            #遍历根据pattern split得到的doc
            for text_piece in text_pieces:
                words = re.finditer(PAT,text_piece)
                #遍历一个Doc里的words
                for word in words:
                    str_word = word.group(0)
                    bytes_of_word = str_word.encode("utf-8")

                    byte_token_tuple = tuple(bytes([byte_value]) for byte_value in bytes_of_word)
                    frequency_table[byte_token_tuple]+=1

        #根据vocabulary的大小得到需要merge的次数
        times_to_merge = vocab_size-len(Vocabulary)
        
        
            # frequency_table_for_merge = Counter()

            # for byte_token_tuple in frequency_table:
            #     #先统计当前的相邻bytes pair的frequency
            #     length = len(byte_token_tuple)
            #     for j in range(0, length - 1):
            #         new_byte_token_tuple = tuple(byte_value for byte_value in byte_token_tuple[j:j + 2])
            #         frequency_table_for_merge[new_byte_token_tuple] += frequency_table[byte_token_tuple]

            # #找到当前最大的frequency
            # if not frequency_table_for_merge:
            #     break
            # biggest_tuple = next(iter(frequency_table_for_merge))
            # for pair in frequency_table_for_merge:
            #     count = frequency_table_for_merge[pair]
            #     current_biggest_count = frequency_table_for_merge[biggest_tuple]
            #     if count > current_biggest_count:
            #         biggest_tuple = pair
            #     if count == current_biggest_count:
            #         if pair > biggest_tuple:
            #             biggest_tuple = pair

            #merge，更新frequency_table,vocabulary,

            #优化版merge
        items = list(frequency_table.items())
        #记录每一个word
        word_tokens = [tokens for tokens,_ in items]
        #记录每一个word的freq
        word_freqs = [frequency for _,frequency in items]

        pair_counts = Counter()
        pair_to_words = defaultdict(set)

        for word_id,tokens in enumerate(word_tokens):
            word_freqency = word_freqs[word_id]
            local_pairs = Counter(pairwise(tokens))

            for pair,occurrence_count in local_pairs.items():
                pair_counts[pair]+=occurrence_count*word_freqency
                pair_to_words[pair].add(word_id)

        for merge_round in range(times_to_merge):
            if not pair_counts:
                break

            best_pair = max(
                pair_counts,
                key = lambda pair: (pair_counts[pair],pair)
            )

            affected_word_ids = list(pair_to_words[best_pair])
            merged_token = best_pair[0]+best_pair[1]

            ##1. 减掉这个词原来的 pair 统计
            for word_id in affected_word_ids:
                old_tokens = word_tokens[word_id]
                word_freqency = word_freqs[word_id]

                old_local_pairs = Counter(pairwise(old_tokens))

                for pair,occurrence_count in old_local_pairs.items():
                    contribution = occurrence_count*word_freqency
                    pair_counts[pair] -= contribution

                    word_ids = pair_to_words[pair]
                    word_ids.discard(word_id)

                    if pair_counts[pair] == 0:
                        del pair_counts[pair]

                    if not word_ids:
                        del pair_to_words[pair]

                # 2 在word_tokens[word_id]中执行merge
                new_tokens = []
                position = 0
                while position < len(old_tokens):
                    if position+1<len(old_tokens) and (old_tokens[position],old_tokens[position+1]) == best_pair:
                        new_tokens.append(merged_token)
                        position+=2
                    else:
                        new_tokens.append(old_tokens[position])
                        position+=1

                new_tokens = tuple(new_tokens)
                word_tokens[word_id] = new_tokens

                #3 保存新的word_tokens[word_id]
                #4 把新的pair 统计加回来
                new_local_pairs = Counter(pairwise(new_tokens))
                word_freqency = word_freqs[word_id]

                for pair,occurrence_count in new_local_pairs.items():
                    contribution = occurrence_count*word_freqency
                    pair_counts[pair]+=contribution
                    pair_to_words[pair].add(word_id)
                    
            merges.append(best_pair)

            # new_frequency_table = Counter()
            # merged_token = biggest_tuple[0] + biggest_tuple[1]
            # merges.append((biggest_tuple[0],biggest_tuple[1]))

            # for byte_token_tuple, count in frequency_table.items():
            #     merged_tokens = []
            #     position = 0

            #     while position < len(byte_token_tuple):
            #         if position + 1 < len(byte_token_tuple) and byte_token_tuple[position:position + 2] == biggest_tuple:
            #             merged_tokens.append(merged_token)
            #             position += 2
            #         else:
            #             merged_tokens.append(byte_token_tuple[position])
            #             position += 1

            #     new_frequency_table[tuple(merged_tokens)] += count

            # frequency_table = new_frequency_table

        ##所有merge次数用完
        #将table中的element加入vocabulary
        vocab_values = set(Vocabulary.values())

    for left_token, right_token in merges:
        merged_token = left_token + right_token

        assert merged_token not in vocab_values, f"重复 token: {merged_token!r}"

        Vocabulary[current_index] = merged_token
        vocab_values.add(merged_token)
        current_index += 1

    return (Vocabulary,merges)


    