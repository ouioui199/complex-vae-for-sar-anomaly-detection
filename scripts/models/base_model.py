from abc import ABC
from argparse import Namespace
from typing import Sequence
from pathlib import Path
import os

import torch
from torch import Tensor
import numpy as np

from lightning import LightningModule
from torchvision.utils import make_grid

from models.utils import MinMaxDenormalize


class BaseModel(LightningModule, ABC):
    
    def __init__(self, opt: Namespace, image_out_dir: str):
        super().__init__()

        self.opt = opt
        self.image_save_dir = Path(f'{image_out_dir}')

        self.denorm = MinMaxDenormalize
    
    def get_name_ext(self, filepath: str, add_epoch: bool = True) -> Sequence[str]:
        filename = os.path.basename(filepath)
        name, ext = os.path.splitext(filename)
        if add_epoch:
            name = f'{name}_epoch_{self.current_epoch}'
        
        return name, ext
    
    def log_image(self, label: str, data: Tensor) -> None:
        grid = make_grid(data)
        self.loggers[0].experiment.add_image(
            label, grid, self.current_epoch
        )
        
    def compute_scm_smv(self, x: Tensor) -> Sequence[Tensor]: 
        (_, _, N) = x.shape
        
        sigma = torch.bmm(x, x.conj().transpose(-2, -1)) / (N-1)
        mu = None
            
        return sigma, mu
    
    def create_manhattan_anomaly_map(self, pred: np.memmap, input: np.memmap) -> np.memmap:
        difference = np.abs(pred - input)
        return difference, difference.sum(axis=0)
    
    def create_euclidean_anomaly_map(self, pred: np.memmap, input: np.memmap) -> np.memmap:
        difference = (pred - input) ** 2
        return np.sqrt(difference), np.sqrt(difference.sum(axis=0))
