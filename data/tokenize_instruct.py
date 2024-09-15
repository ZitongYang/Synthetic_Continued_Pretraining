import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from typing import Dict
from functools import partial
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset


def process(example: Dict, tokenizer: AutoTokenizer)->Dict:
    """
    Tokenize the text and return the tokenized text
    """
    ids = tokenizer.apply_chat_template(example['messages'])
    return dict(ids=ids,len=len(ids))

def write_to_memmap(dset: Dataset, filename: str):
    dtype = np.int32
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = min(1024, len(dset))
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        # Batch together samples for faster write
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        # Write into mmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
        arr.flush()

def tokenize_and_save(tokenizer: AutoTokenizer):
    """
    After saving the tokenized text, we may read them as
    >>> import numpy as np
    >>> arr = np.memmap('data/dataset/bins/ultrachat_test.bin', mode='r', dtype=np.int32)
    >>> len(arr)
    27683545
    >>> arr[:5]
    memmap([128000, 128006,   9125, 128007,    271], dtype=int32)
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', use_fast=True)
    >>> print(tokenizer.decode(arr[11000000:11000010]))
    Force, flexible days and times. Contact: Cheryl
    """
    process_map = partial(process, tokenizer=tokenizer)
    # loading dataset
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k",
                           trust_remote_code=True)

    filename = f'data/dataset/bins/ultrachat'
    # core tokenization operation happening
    tokenized_train = dataset['train_sft'].map(process_map,
                                           remove_columns=dataset['train_sft'][0].keys(),
                                           desc='Tokenizing training split',
                                           num_proc=16)
    tokenized_test = dataset['test_sft'].map(process_map,
                                         remove_columns=dataset['train_sft'][0].keys(),
                                         desc='Tokenizing test split',
                                         num_proc=16)

    # concatenate all the ids in each dataset into one large file we can use for training
    write_to_memmap(tokenized_train, f"{filename}_train.bin")
    write_to_memmap(tokenized_test, f"{filename}_test.bin")


if __name__ == '__main__':
    # loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', use_fast=True)
    tokenizer.model_max_length=2**20 # this is to hide the token_len>2048 wraning
    # tokenizing the dataset
    tokenize_and_save(tokenizer)