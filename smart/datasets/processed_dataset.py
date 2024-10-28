import os
import mmap
import pickle
from typing import Callable, List, Optional, Tuple, Union
import pandas as pd
from torch_geometric.data import Dataset
from smart.utils.log import Logging
import numpy as np
from .preprocess import TokenProcessor


class ProcessedDataset(Dataset):
    def __init__(self,
                 root: str,
                 split: str,
                 raw_dir: List[str] = None,
                 processed_dir: List[str] = None,
                 transform: Optional[Callable] = None,
                 dim: int = 3,
                 num_historical_steps: int = 50,
                 num_future_steps: int = 60,
                 predict_unseen_agents: bool = False,
                 vector_repr: bool = True,
                 cluster: bool = False,
                 processor=None,
                 use_intention=False,
                 token_size=512) -> None:
        self.logger = Logging().log(level='DEBUG')
        self.logger.debug("Starting loading dataset with ProcessedDataset")
        
        if split not in ('train', 'val', 'test'):
            raise ValueError(f"{split} is not a valid split")
        self.split = split
        
        if processed_dir is None:
            raise ValueError(f"add processed_dir in processed dataset")
        
        self._processed_dir = processed_dir
        
        self._processed_paths = []
        for processed_dir in self._processed_dir:
            processed_dir = os.path.expanduser(os.path.normpath(processed_dir))
            file_list = os.listdir(processed_dir)
            self._processed_paths.extend([os.path.join(processed_dir, f) for f in file_list])
        
        self._num_samples = len(self._processed_paths)
        
        self.logger.debug("The number of {} dataset is ".format(split) + str(self._num_samples))
        super().__init__(root=root, transform=transform, pre_transform=None, pre_filter=None)

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def len(self) -> int:
        return self._num_samples

    def generate_ref_token(self):
        pass

    def get(self, idx: int):
        with open(self.processed_paths[idx], 'rb') as handle:
            data = pickle.load(handle)
        return data
    
    # def get(self, idx: int):
    #     with open(self.processed_paths[idx], 'rb') as f:
    #         # 使用内存映射方式读取文件
    #         mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    #         data = pickle.load(mmapped_file)
    #         mmapped_file.close()
    #     return data

