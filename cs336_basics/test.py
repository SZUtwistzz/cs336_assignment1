from pretokenization_example import find_chunk_boundaries
import regex as re
from collections import Counter

input_path = "E:/cs336/assignment1-basics/cs336_basics/test.txt"
special_token = "<|endoftext|>"

with open(input_path, "rb") as f:
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    frequency_table = Counter()
    valueOfEscape = re.escape(special_token)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        ##pattern = "|".join(special_tokens) 只有一个special_token不需要join
        text_pieces = re.split(valueOfEscape,chunk)
        
        for text_piece in text_pieces:
            words = re.finditer(PAT,text_piece)
            for word in words:
                str_word = word.group(0)
                bytes_of_word = str_word.encode("utf-8")

                byte_token_tuple = tuple(bytes([byte_value]) for byte_value in bytes_of_word)
                frequency_table[byte_token_tuple]+=1

    # for frequency in frequency_table:
    #     print(frequency)

    times_to_merge = 6
    
    #一轮merge
    for i in range(times_to_merge):
        frequency_table_for_merge = Counter()

        for byte_token_tuple in frequency_table:
                #先统计当前的相邻bytes pair的frequency
                length = len(byte_token_tuple)
                for i in range(0,length-1):
                    new_byte_token_tuple = tuple(byte_value for byte_value in byte_token_tuple[i:i+2])
                    frequency_table_for_merge[new_byte_token_tuple]+=frequency_table[byte_token_tuple]
        
            #找到当前最大的frequency
                if not frequency_table_for_merge:
                    break
                biggest_tuple = next(iter(frequency_table_for_merge))
                for pair in frequency_table_for_merge:
                    count = frequency_table_for_merge[pair]
                    current_biggest_count = frequency_table_for_merge[biggest_tuple]
                    if count>current_biggest_count:
                        biggest_tuple = pair
                    if count == current_biggest_count:
                        if pair>biggest_tuple:
                            biggest_tuple = pair
            
                #merge，更新frequency_table,vocabulary,
                new_frequency_table = Counter()
                merged_token = biggest_tuple[0]+biggest_tuple[1]

                for byte_token_tuple, count in frequency_table.items():
                    merged_tokens = []
                    position = 0 

                    while position < len(byte_token_tuple):
                        if position+1 < len(byte_token_tuple) and byte_token_tuple[position:position+2] == biggest_tuple:
                            merged_tokens.append(merged_token)
                            position+=2

                        else:
                            merged_tokens.append(byte_token_tuple[position])
                            position+=1

                    new_frequency_table[tuple(merged_tokens)] += count

        frequency_table = new_frequency_table

    for item in frequency_table:
        print(item) 
    # # for pair,count in frequency_table_for_merge.items():
    # #      print(pair,count)