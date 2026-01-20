from typing import Any, Dict
from argparse import ArgumentParser
from abc import ABC, abstractmethod
import os, glob

import numpy as np
from torch.utils.data import Dataset
from torchcvnn.transforms import ToTensor

import datasets.utils as D_U
import datasets.transforms as T


class ReconstructorFile(ABC):

    def __init__(
        self, 
        opt: ArgumentParser,
        filepath: str,
        phase: str
    ) -> None:
        self.opt = opt
        self.filepath = filepath
        self.phase = phase
        # Load image
        self.data = np.load(filepath, mmap_mode='r')
        self.data = np.abs(self.data) if 'cplx' not in opt.recon_model else self.data            
        # Get min and max for normalization
        self.min_val, self.max_val = self.get_min_max() if phase == 'train' else np.array(self.opt.normalization_values)
        if len(self.min_val) != len(self.max_val):
            raise ValueError("Normalization values length mismatch between min and max.")
        if len(self.min_val) != self.data.shape[0]:
            raise ValueError("Normalization values length does not match number of channels in the data.")
        self.min_val, self.max_val = self.min_val.reshape(-1, 1, 1), self.max_val.reshape(-1, 1, 1)
        # Ensure CHW format
        self.data = D_U.ensure_chw_format(self.data)
        self.data = self.get_train_valid('data')

    def get_train_valid(self, data_name: str) -> Dict[str, np.ndarray]:
        _, h, w = self.data.shape
        train_valid_threshold = int(max(h, w) * self.opt.train_valid_ratio[0])
        data = getattr(self, data_name)
        axis = 1 if h > w else 2
        train_valid = {
            'train': np.take(data, range(train_valid_threshold), axis=axis),
            'valid': np.take(data, range(train_valid_threshold, data.shape[axis]), axis=axis),
            'predict': data
        }
        
        for (k, v) in train_valid.items():
            if k == self.phase:
                return v
                
    def get_min_max(self) -> Dict[str, float]:
        data = self.data if not np.iscomplexobj(self.data) else np.abs(self.data)
        data = np.log(data + np.spacing(1))
        min_val, max_val = data.min(axis=(1,2)), data.max(axis=(1,2))
        if hasattr(self.opt, 'recon_train_slc') and self.opt.recon_train_slc:
            min_val, max_val = np.percentile(data, (5, 95), axis=(1,2))
        return min_val, max_val
    
    def __len__(self) -> int:
        return self.nsamples_per_rows * self.nsamples_per_cols
    
    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        pass


class BaseDataset(ABC, Dataset):
    def __init__(
            self, 
            opt: ArgumentParser,
            phase: str
    ) -> None:
        super().__init__()

        assert phase in ['train', 'valid', 'predict', 'test'], "Dataset only accept 'train', 'valid', 'predict' as phase"
        assert sum(opt.train_valid_ratio) == 1., "Train and valid proportion does not cover all image"
        
        self.opt = opt
        self.data_dir = os.path.join(opt.datadir, opt.data_band + '_band') if opt.data_band else opt.datadir
        self.data_dir = os.path.join(self.data_dir, 'train' if phase in ['train', 'valid'] else 'predict')
        self.normalize = T.MinMaxNormalize
        self.to_tensor = ToTensor('float32') if 'cplx' not in opt.recon_model else ToTensor('complex64')
    
    def get_normalization(self) -> None:
        self.dataset_min = np.min([i.min_val for i in self.dataset], axis=0)
        self.dataset_max = np.max([i.max_val for i in self.dataset], axis=0)

        self.normalize = T.MinMaxNormalize(self.dataset_min, self.dataset_max)

    @abstractmethod
    def __len__(self) -> int:
        return 0

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        pass


class ReconstructorDataset(BaseDataset):
    
    def __init__(
        self, 
        opt: ArgumentParser,
        phase: str
    ) -> None:
        super().__init__(opt, phase)
        
        self.data_dir = os.path.join(self.data_dir, 'slc')
        if D_U.is_directory_empty(self.data_dir):
            raise FileNotFoundError(f"data slc directory is empty or doesn't exist.")
        
        filepath = glob.glob(f'{self.data_dir}/*.npy')
        self.filepath = sorted([s for s in filepath if (opt.data_band in s) and ('Combine' in s)])
            
    def __len__(self) -> int:
        return sum([len(image) for image in self.dataset])
    