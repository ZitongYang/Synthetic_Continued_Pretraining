from torch.utils.data import Dataset
from typing import Dict
import numpy as np
import torch

def _get_bin(task_name: str, split: str):
    assert task_name in ['quality', 'rehersal', 'instruct']
    bin_data_dir = 'data/dataset/bins'
    implemented_quality_split = {
        'entigraph': f'{bin_data_dir}/quality_all-entigraphgpt-4-turbo.bin',
    }
    implemented_rehersal_split = {
        'rpj-train': f'{bin_data_dir}/togethercomputer_RedPajama_Data_1T_Sample_train.bin',
        'rpj-test': f'{bin_data_dir}/togethercomputer_RedPajama_Data_1T_Sample_test.bin'
    }
    implemented_instruct_split = {
        'ultrachat-train': f'{bin_data_dir}/ultrachat_train.bin',
        'ultrachat-test': f'{bin_data_dir}/ultrachat_test.bin'
    }
    if task_name == 'quality':
        assert split in implemented_quality_split
        return implemented_quality_split[split]
    elif task_name == 'rehersal':
        assert split in implemented_rehersal_split
        return implemented_rehersal_split[split]
    elif task_name == 'instruct':
        assert split in implemented_instruct_split
        return implemented_instruct_split[split]
    else:
        raise NotImplementedError(f"Task {task_name} is not implemented")


class _MemmapDataset(Dataset):
    def __init__(self, block_size: int, bin_file: str, subsample_ratio: float):
        self.block_size = block_size
        self.ids = np.memmap(bin_file, dtype=np.int32, mode='r')
        self.ids = self.ids[:int(len(self.ids)*subsample_ratio)]

    def __len__(self):
        return int(len(self.ids)/self.block_size)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        assert i < len(self)
        start_ind = i*self.block_size
        end_ind = (i+1)*self.block_size
        x_id = self.ids[start_ind:end_ind].copy()
        return dict(input_ids=torch.from_numpy(x_id).long(),
                    labels=torch.from_numpy(x_id).long())

class CPTDataset(_MemmapDataset):
    def __init__ (self, block_size: int, rehersal_rate: float,
                 subsample_ratio: float):
        assert rehersal_rate <= 1.0
        self.rehersal_rate = rehersal_rate
        self.rehersal_data = _MemmapDataset(block_size, _get_bin('rehersal', 'rpj-train'), 1.0)
        super().__init__(block_size,
                            _get_bin('quality', 'entigraph'),
                            subsample_ratio)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        if np.random.rand() < self.rehersal_rate:
            idx = np.random.randint(len(self.rehersal_data))
            return self.rehersal_data[idx]
        else:
            return super().__getitem__(i)

def get_task_data_module(task_name: str,
                         block_size: int,
                         rehersal_rate: float,
                         subsample_ratio: float,
                         **kwargs):
    if task_name == 'quality':
        train = CPTDataset(block_size, rehersal_rate, subsample_ratio)
        val = _MemmapDataset(block_size, _get_bin('rehersal', 'rpj-test'), 1.0)
        return dict(train_dataset=train, eval_dataset=val)
    if task_name == 'instruct':
        train = _MemmapDataset(block_size, _get_bin('instruct', 'ultrachat-train'), 1.0)
        val = _MemmapDataset(block_size, _get_bin('instruct', 'ultrachat-test'), 1.0)
        return dict(train_dataset=train, eval_dataset=val)
    else:
        raise NotImplementedError(f"Task {task_name} is not implemented")


if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)
    tokenizer.model_max_length=2**20 # this is to hide the token_len>128K wraning

    block_size = 2048
    rehersal_rate = 0.1
    subsample_ratio = 1.0
    task_name = 'quality'
    data_module = get_task_data_module(task_name, block_size,
                                       rehersal_rate, subsample_ratio)
    for example in data_module['train_dataset']:
        print(tokenizer.decode(example['input_ids']))
        import pdb; pdb.set_trace()