from typing import Any, Dict
from argparse import ArgumentParser
from abc import ABC, abstractmethod
import os, glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchcvnn.transforms import ToTensor
from torchcvnn.transforms.functional import equalize

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
        if 'DSO' in filepath:
            self.data, _, _ = D_U.process_dot_mat(opt, filepath, np)
        else:
            self.data = np.load(filepath, mmap_mode='r')

        self.data = D_U.process_image_representation(self.data, np, opt)
        if hasattr(self.opt, 'recon_classification_guided') and self.opt.recon_classification_guided and phase == 'train':
            rx_path = filepath.replace('Combined', 'RX-SCM-shift')
            if not os.path.exists(rx_path):
                raise FileNotFoundError(f"RX map file not found at {rx_path}. Please compute the RX map before using classification guided reconstruction.")
            
            self.data_rx = np.load(rx_path, mmap_mode='r')
            self.data = self.data[
                :, 
                opt.rx_box_car_size // 2 : -opt.rx_box_car_size // 2, 
                opt.rx_box_car_size // 2 : -opt.rx_box_car_size // 2
            ]
            assert self.data.shape[1:] == self.data_rx.shape[1:], f"Image and RX map shape mismatch: {self.data.shape} != {self.data_rx.shape}"
            self.data_rx = D_U.set_probability_false_alarm(np.log(self.data_rx), 0.05, binarize=True)[0]
            
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
        self.data_rx = self.get_train_valid('data_rx') if hasattr(self, 'data_rx') else None

        if hasattr(self, 'label'):
            self.label = self.get_train_valid('label')

    def get_train_valid(self, data_name: str) -> Dict[str, np.ndarray]:
        data = getattr(self, data_name)
        _, h, w = data.shape
        train_valid_threshold = int(max(h, w) * self.opt.train_valid_ratio[0])
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
        self.to_tensor = ToTensor('complex64')
    
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
    