import os, ast
from argparse import ArgumentParser, Namespace
from typing import List
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from lightning import LightningDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from lightning.pytorch.utilities import rank_zero_only
from torch.utils.data import DataLoader
from torchcvnn.transforms.functional import equalize

import datasets.dataset as D
import datasets.transforms as T


class TBLogger(TensorBoardLogger):
    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop('epoch', None)
        metrics = {k: v for k, v in metrics.items() if ('step' not in k) and ('val' not in k)}
        return super().log_metrics(metrics, step)
    
    
class CustomProgressBar(TQDMProgressBar):
    
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items
    
    def init_train_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for training."""
        bar = super().init_train_tqdm()
        bar.ascii = ' >'
        return bar
    
    def init_validation_tqdm(self) -> Tqdm:
        bar = super().init_validation_tqdm()
        bar.ascii = ' >'
        return bar
    
    def init_predict_tqdm(self) -> Tqdm:
        bar = super().init_predict_tqdm()
        bar.ascii = ' >'
        return bar
    
    def init_test_tqdm(self) -> Tqdm:
        bar = super().init_test_tqdm()
        bar.ascii = ' >'
        return bar


class BaseDataModule(LightningDataModule):
    def __init__(self, opt: Namespace) -> None:
        super().__init__()
        self.opt = opt
        self.pin_memory = False

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.train_batch_size,
            num_workers=self.opt.workers,
            pin_memory=True,
            shuffle=True,
            persistent_workers=True,
            prefetch_factor=100
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.opt.workers,
            # pin_memory=True,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=100
        )
        
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.pred_dataset, 
            batch_size=self.pred_batch_size,
            num_workers=self.opt.workers,
            shuffle=False,
            persistent_workers=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.opt.recon_test_batch_size,
            num_workers=self.opt.workers,
            shuffle=False,
            persistent_workers=True
        )


class ReconstructionDatasetModule(BaseDataModule):
    def __init__(self, opt: Namespace) -> None:
        super().__init__(opt)

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.train_batch_size = self.opt.recon_train_batch_size
            self.val_batch_size = self.opt.recon_val_batch_size
            self.train_dataset = D.ReconstructorTrainDataset(self.opt)
            self.valid_dataset = D.ReconstructorValidDataset(self.opt)
        
        if stage == 'predict':
            self.pred_batch_size = self.opt.recon_pred_batch_size
            self.pred_dataset = D.ReconstructorPredictDataset(self.opt)
        
        if stage == 'test':
            self.test_dataset = D.ReconstructorTestDataset(self.opt)


class ArgumentParsing:
    
    def __init__(self, parser: ArgumentParser) -> None:
        self.parser = parser
        self.common_args()
        self.train_reconstructor_group = self.parser.add_argument_group()
        self.predict_reconstructor_group = self.parser.add_argument_group()
        self.compute_tsne_group = self.parser.add_argument_group()
        
    def common_args(self) -> None:
        self.parser.add_argument('--version', type=str, required=True)
        self.parser.add_argument('--workers', type=int, default=4)
        self.parser.add_argument('--datadir', type=str, required=True)
        self.parser.add_argument('--train_valid_ratio', type=List, default=[0.8, 0.2])
        self.parser.add_argument('--data_band', type=str, default=None)
        self.parser.add_argument('--rx_type', type=str, default='scm', choices=['scm', 'tyler'])
        self.parser.add_argument('--rx_real_valued', action='store_true')
        self.parser.add_argument('--rx_box_car_size', type=int, default=39)
        self.parser.add_argument('--rx_exclusion_window_size', type=int, default=31)
        self.parser.add_argument('--normalization_values', default=[[2.18594766, 1.68413746, 1.74545192, 2.11463547], [4.92781165, 4.33335114, 4.37936831, 4.89514112]], type=ast.literal_eval, help='Min and max values for min-max normalization. Both should be store in lists, e.g., --normalization_values [[min1, min2, ...], [max1, max2, ...]]')

    def train_reconstructor_args(self, group: ArgumentParser) -> None:
        group.add_argument('--recon_visualize', action='store_true')
        group.add_argument('--recon_train_slc', action='store_true')
        group.add_argument('--recon_patch_size', type=int, default=32)
        group.add_argument('--recon_stride', type=int, default=16)
        group.add_argument('--recon_in_channels', type=int, default=4)
        group.add_argument('--recon_train_batch_size', type=int, default=128)
        group.add_argument('--recon_val_batch_size', type=int, default=512)
        group.add_argument('--recon_epochs', type=int, default=100)
        group.add_argument('--recon_latent_compression', type=int, default=1)
        group.add_argument('--recon_lr_ae', type=float, default=1e-3)
        
        group.add_argument('--kld_weight', type=float, default=1e-4)
        group.add_argument('--recon_regulate_beta', action='store_true')
        group.add_argument('--beta_warmup_type', type=str, default='linear', choices=['linear', 'cyclical'])
        group.add_argument('--beta_n_epochs', type=int, default=None)
        group.add_argument('--recon_beta_start', type=float, default=0.)
        group.add_argument('--recon_beta_end', type=float, default=1.)
        group.add_argument('--recon_beta_proportion', type=float, default=.8)
        group.add_argument('--recon_beta_warmup_epochs', type=int, default=5)
        
        group.add_argument('--recon_classification_guided', action='store_true')
        group.add_argument('--recon_ckpt_path', type=str, default=None, help='Path to a checkpoint to resume training from. If None, training starts from scratch.')
        
    def predict_reconstructor_args(self, group: ArgumentParser) -> None:
        group.add_argument('--recon_train_slc', action='store_true')
        group.add_argument('--recon_predict', action='store_true')
        group.add_argument('--recon_pred_batch_size', type=int, default=128)
        group.add_argument('--recon_test_batch_size', type=int, default=16384)
        group.add_argument('--recon_latent_compression', type=int, default=1)
        
        group.add_argument('--recon_data_prediction', type=str, required=True, choices=['full', 'sample_only', 'valid_only', 'synthetic_only'])
        group.add_argument('--recon_in_channels', type=int, default=4)
        group.add_argument('--recon_anomaly_kernel', type=int, default=11)
        group.add_argument('--recon_patch_size', type=int, default=32)
        group.add_argument('--recon_stride', type=int, default=16)

    def compute_tsne_args(self, group: ArgumentParser) -> None:
        group.add_argument('--tsne_patch_size', type=int, default=64)
        group.add_argument('--tsne_stride', type=int, default=20)
        group.add_argument('--tsne_batch_size', type=int, default=512)
        group.add_argument('--tsne_dataset_min', type=float, default=0.)
        group.add_argument('--tsne_dataset_max', type=float, default=1.)
        group.add_argument('--recon_in_channels', type=int, default=4)
        group.add_argument('--recon_patch_size', type=int, default=32)


def visualize_recon(dataloader: DataLoader, outpath: str) -> None:
    denorm = T.MinMaxDenormalize
    for data in tqdm(dataloader):
        for i in range(len(data['filepath'])):
            filepath = data['filepath'][i]
            (pos_row, pos_col) = (data['row'][i], data['col'][i]) if 'row' in data else (0, 0)
            data_min, data_max = data['min'][i], data['max'][i]
            image_denorm = denorm(data_min, data_max)(data['image'][i]).numpy()

            image = equalize(image_denorm.transpose(1, 2, 0).squeeze(), plower=0, pupper=100)
            image = np.stack([image[:, :, 0], image[:, :, 1], image[:, :, 3]], axis=2) # Convert to RGB
            image = Image.fromarray(image)

            filename = os.path.basename(filepath)
            name, ext = os.path.splitext(filename)
            filename = f'{name}_{pos_row}_{pos_col}'
            image.save(Path(f'{outpath}/{filename}.png'))

            np.save(Path(f'{outpath}/{filename}{ext}'), image_denorm)
