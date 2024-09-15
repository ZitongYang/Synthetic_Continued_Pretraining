import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from typing import Optional, Dict
from dataclasses import dataclass, field
from functools import partial
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, HfArgumentParser
from datasets import load_dataset, Dataset

@dataclass
class TokenizeConfig:
    tokenizer_model_name: Optional[str] = field(default='meta-llama/Meta-Llama-3-8B')
    load_dataset_path: Optional[str] = field(default='togethercomputer/RedPajama-Data-1T-Sample')
    num_proc: Optional[int] = field(default=16)
    # logging
    wandb_logging: Optional[bool] = field(default=False)
    wandb_project: Optional[str] = field(default='synthetic-continued-pretraining')
    wandb_run_name: Optional[str] = field(default='tokenizing-redpj')

    def __post_init__(self):
        if self.load_dataset_path == 'togethercomputer/RedPajama-Data-1T':
            self.text_key = 'text'
        elif self.load_dataset_path == 'togethercomputer/RedPajama-Data-1T-Sample':
            self.text_key = 'text'
        else:
            raise NotImplementedError('Only support RedPajama-Data-1T and RedPajama-Data-V2')


def process(example: Dict, tokenizer: AutoTokenizer, text_key: str)->Dict:
    """
    Tokenize the text and return the tokenized text
    """
    ids = tokenizer.encode(example[text_key]) # add_special_tokens=True to add BOS token
    ids.append(tokenizer.eos_token_id) # add the end of text token
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


def tokenize_and_save(tokenizer: AutoTokenizer, config: TokenizeConfig):
    """
    After saving the tokenized text, we may read them as
    >>> import numpy as np
    >>> arr = np.memmap('data/dataset/bins/togethercomputer_RedPajama_Data_1T_Sample_None.bin', mode='r', dtype=np.int32)
    >>> len(arr)
    1142631027
    >>> arr[:5]
    memmap([128000,  13379,    374,  40467,  33350], dtype=int32)
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B', use_fast=True)
    >>> print(tokenizer.decode(arr[11000000:11000010]))
    ycat Dolls creator. Kim got up on
    """
    process_map = partial(process, tokenizer=tokenizer, text_key=config.text_key)
    # loading dataset
    dataset = load_dataset(config.load_dataset_path,
                           trust_remote_code=True)
    dataset = dataset['train']
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.train_test_split(test_size=0.01)

    filename = f'data/dataset/bins/{config.load_dataset_path.replace("/", "_").replace("-","_")}'
    # core tokenization operation happening
    tokenized_train = dataset['train'].map(process_map,
                                           remove_columns=dataset['train'][0].keys(),
                                           desc='Tokenizing training split',
                                           num_proc=config.num_proc)
    tokenized_test = dataset['test'].map(process_map,
                                         remove_columns=dataset['train'][0].keys(),
                                         desc='Tokenizing test split',
                                         num_proc=config.num_proc)

    # concatenate all the ids in each dataset into one large file we can use for training
    write_to_memmap(tokenized_train, f"{filename}_train.bin")
    write_to_memmap(tokenized_test, f"{filename}_test.bin")


if __name__ == '__main__':
    # parsing input
    parser = HfArgumentParser(TokenizeConfig)
    config = parser.parse_args_into_dataclasses()[0]
    # logging
    if config.wandb_logging:
        import wandb
        wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=config)
    # loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_model_name, use_fast=True)
    tokenizer.model_max_length=2**20 # this is to hide the token_len>2048 wraning
    # tokenizing the dataset
    tokenize_and_save(tokenizer, config)