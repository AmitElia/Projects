from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from scipy.spatial.distance import pdist
from data import data_constants

from typing import List, Dict

get_value_from_dict = np.vectorize(lambda key, d: d.get(key))

class Spice2Dataset(Dataset):
    def __init__(self, npz_file_path: str):
        self.file_path = npz_file_path
        self.npz_file = np.load(self.file_path, allow_pickle=True)
        self.key_list = list(self.npz_file.keys())
    
    def __len__(self):
        
        return len(self.key_list)

    def __getitem__(self, index):
        return self.npz_file[self.key_list[index]].item()

def pad_1d(samples: List[np.array], total_len: int, pad_value=0) -> torch.Tensor:
    #adapted from Uni-Mol
    batch_size = len(samples)
    tensor = torch.full([batch_size, total_len], pad_value, dtype=samples[0].dtype)
    for i in range(batch_size):
        tensor[i, : samples[i].shape[0]] = samples[i]
    return tensor

def pad_1d_feat(samples: List[np.array], total_len: int, pad_value=0) -> torch.Tensor:
    #adapted from Uni-Mol
    batch_size = len(samples)
    assert len(samples[0].shape) == 2
    feat_size = samples[0].shape[-1]
    tensor = torch.full(
        [batch_size, total_len, feat_size], pad_value, dtype=samples[0].dtype
    )
    for i in range(batch_size):
        tensor[i, : samples[i].shape[0]] = samples[i]
    return tensor

def pad_2d(samples: List[np.array], total_len: int, pad_value=0) -> torch.Tensor:
    #adapted from Uni-Mol
    batch_size = len(samples)
    tensor = torch.full(
        [batch_size, total_len, total_len], pad_value, dtype=samples[0].dtype
    )
    for i in range(batch_size):
        tensor[i, : samples[i].shape[0], : samples[i].shape[1]] = samples[i]
    return tensor

def collate_molecular_data(data: Dict[str, np.array]) -> Dict[str, torch.Tensor]:
    pad_function_dict = {'chiral_mask': lambda d, l: pad_1d(d, l, pad_value=False).int(),
                         'atomic_numbers': lambda d, l: pad_1d(d, l, pad_value=data_constants.atomic_number_mask_index).long(), 
                         'charges': pad_1d,
                         'chiral_center_neighbors': lambda d, l: pad_1d_feat(d, l, pad_value=-1), 
                         'conformation': pad_1d_feat, 
                         'topological_dist': lambda d, l: pad_2d(d, l, pad_value=-1),  
                         'dist': lambda d, l: pad_2d(d, l, pad_value=-1).float(),
                         }
    key_list = data[0].keys()
    collated = {}
    total_len = max([datum['atomic_numbers'].shape[0] for datum in data])
    for k in key_list:
        if k not in pad_function_dict: #means this data doesn't need to be collated
            continue
        if k == 'atomic_numbers': #convert atomic number to indices starting from 0
            to_pad = [torch.tensor(get_value_from_dict(datum[k], data_constants.atomic_number_to_index)).squeeze() for datum in data]
        else:
            to_pad = [torch.tensor(datum[k]).squeeze() for datum in data]
        collated[k] = pad_function_dict[k](to_pad, total_len)
    collated['atom_mask'] = (collated['atomic_numbers'] != data_constants.atomic_number_mask_index).int()
    return collated