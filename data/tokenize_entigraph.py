import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from typing import List
import numpy as np
from transformers import AutoTokenizer
import random
import glob
from tqdm import tqdm
from utils.io_utils import jload

def get_tokenizer(tokenizer_model_name: str)-> AutoTokenizer:
    # loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name, use_fast=True)
    tokenizer.model_max_length=2**20 # this is to hide the token_len>128K wraning
    return tokenizer

def tokenize_list(text_list: List[str]) -> List[int]:
    """
    Tokenize the text and return the tokenized text
    """
    random.shuffle(text_list)
    tokenizer = get_tokenizer("meta-llama/Meta-Llama-3-8B")
    all_ids = []
    for text in tqdm(text_list):
        if text:
            ids = tokenizer.encode(text) # add_special_tokens=True to add BOS token
            ids.append(tokenizer.eos_token_id) # add the end of text token
            all_ids.extend(ids)
    return all_ids

def write_to_memmap_single(ids: List[int], filename: str):
    filename = f'data/dataset/bins/{filename}'
    print(f'Writing to {filename} with length {len(ids)}')
    dtype = np.int32
    ids_arr = np.array(ids, dtype=dtype)
    arr_len = len(ids_arr)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    arr[:] = ids_arr
    arr.flush()

def _glob_all_json(dir_name: str) -> List[str]:
    return glob.glob(f'{dir_name}/*.json') + glob.glob(f'{dir_name}/.*.json')

def _get_quality_graph(model_name: str) -> List[str]:
    files = _glob_all_json(f'data/dataset/raw/quality_entigraph_{model_name}')
    result = []
    for file in files:
        content = jload(file)
        result.extend(content[1:])
    return result

def tokenize_quality_graph(model_name: str):
    quality = _get_quality_graph(model_name)
    write_to_memmap_single(tokenize_list(quality), f'quality_all-entigraph{model_name}.bin')

if __name__ == '__main__':
    # Writing to data/dataset/bins/quality_all-graphgpt-4-turbo.bin with length 599385906 (599M)
    tokenize_quality_graph('gpt-4-turbo')