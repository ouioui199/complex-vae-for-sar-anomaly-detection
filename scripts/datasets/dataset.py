from typing import Dict
from argparse import ArgumentParser
from overrides import override
import os, glob

import torch
import numpy as np
from torch import Tensor

from datasets.base_dataset import ReconstructorFile, ReconstructorDataset, BaseDataset


class ReconstructorTrainFile(ReconstructorFile):
    
    def __init__(
        self, 
        opt: ArgumentParser,
        filepath: str,
        phase: str = 'train',
    ) -> None:
        super().__init__(opt, filepath, phase)

        _, nrows, ncols = self.data.shape
        self.nsamples_per_rows = (ncols - opt.recon_patch_size) // opt.recon_stride + 1
        self.nsamples_per_cols = (nrows - opt.recon_patch_size) // opt.recon_stride + 1
    
    @override
    def __getitem__(self, index: int) -> Dict[str, np.memmap | str | int]:
        row = index // self.nsamples_per_rows
        col = index % self.nsamples_per_rows

        row_start = row * self.opt.recon_stride
        col_start = col * self.opt.recon_stride

        output = {
            'data': self.data[:, row_start : row_start + self.opt.recon_patch_size, col_start : col_start + self.opt.recon_patch_size],
            'filepath': self.filepath
        }
        
        if self.opt.recon_classification_guided:
            output['label'] = self.data_rx[row_start : row_start + self.opt.recon_patch_size, col_start : col_start + self.opt.recon_patch_size]

        if self.opt.recon_visualize:
            output['row'] = row
            output['col'] = col

        return output
    

class ReconstructorValidFile(ReconstructorFile):

    def __init__(
        self, 
        opt: ArgumentParser, 
        filepath: str,
        phase: str = 'valid',
    ) -> None:
        super().__init__(opt, filepath, phase)

        _, h, w = self.data.shape
        if phase == 'valid':
            stride = self.opt.recon_stride // 4
        elif self.opt.recon_data_prediction == 'synthetic_only':
            stride = 1
        if h == self.opt.recon_patch_size:
            self.x_range = list(np.array([0]))
        else:
            self.x_range = list(range(0, h - self.opt.recon_patch_size, stride))
            if (self.x_range[-1] + self.opt.recon_patch_size) < h :
                self.x_range.extend(range(h - self.opt.recon_patch_size, h - self.opt.recon_patch_size + 1))
        
        if w == self.opt.recon_patch_size:
            self.y_range = list(np.array([0]))
        else:
            self.y_range = list(range(0, w - self.opt.recon_patch_size, stride))
            if (self.y_range[-1] + self.opt.recon_patch_size) < w:
                self.y_range.extend(range(w - self.opt.recon_patch_size, w - self.opt.recon_patch_size + 1))

        self.nsamples_per_rows = len(self.y_range)
        self.nsamples_per_cols = len(self.x_range)
    
    @override
    def __getitem__(self, index: int) -> Dict[str, np.memmap | str]:
        row = index % self.nsamples_per_rows
        col = index // self.nsamples_per_rows

        row_start = self.y_range[row]
        col_start = self.x_range[col]

        output = {
            'data': self.data[:, col_start : col_start + self.opt.recon_patch_size, row_start : row_start + self.opt.recon_patch_size],
            'filepath': self.filepath,
            'row': row_start,
            'col': col_start
        }

        return output
    

class ReconstructorTestFile:

    def __init__(
        self, 
        opt: ArgumentParser, 
        filepath_input: str,
        filepath_reconstructed: str
    ) -> None:
        """Return the whole validation part of the image"""
        self.opt = opt
        
        self.input = np.load(filepath_input, mmap_mode='r')
        self.input = np.abs(self.input) if 'cplx' not in opt.recon_model else self.input

        self.reconstructed = np.load(filepath_reconstructed, mmap_mode='r')
        self.reconstructed = np.abs(self.reconstructed) if 'cplx' not in opt.recon_model else self.reconstructed

        _, self.height, self.width = self.input.shape
        self.half_kernel = opt.recon_anomaly_kernel // 2
        
        self.height = self.height - self.half_kernel * 2
        self.width = self.width - self.half_kernel * 2
        
    def __len__(self) -> int:
        """Returns the total number of patches of the dataset."""
        return self.height * self.width
    
    def __getitem__(self, index: int) -> Dict[str, Tensor | str | int]:
        """Returns the index-th patch of the associated NPY file.

        Args:
            index (int): index of the patch to be extracted

        Returns:
            Dict[str, Tensor | str | int]: Tensor and metadata
        """
        col_idx = index % self.height + self.half_kernel
        row_idx = index // self.height + self.half_kernel
        
        up = col_idx - self.half_kernel
        down = col_idx + self.half_kernel + 1
        left = row_idx - self.half_kernel
        right = row_idx + self.half_kernel + 1
        
        input_patch = self.input[:, up:down, left:right]
        reconstructed_patch = self.reconstructed[:, up:down, left:right]
        
        return {
            'input': input_patch,
            'reconstructed': reconstructed_patch,
            'col': col_idx - self.half_kernel,
            'row': row_idx - self.half_kernel
        }


class ReconstructorTrainDataset(ReconstructorDataset):

    def __init__(
            self, 
            opt: ArgumentParser,
            phase: str = 'train'
    ) -> None:
        super().__init__(opt, phase)
        self.filepath = [fp for fp in self.filepath if ('anomaly' not in fp) and ('synthetic' not in fp) and ('crop' not in fp)]
        self.dataset = [ReconstructorTrainFile(opt, filepath, phase) for filepath in self.filepath]
        self.get_normalization()

    @override
    def __getitem__(self, index: int) -> Dict[str, Tensor | str | int]:
        for image in self.dataset:
            if index < len(image):
                break
            index -= len(image)
        
        output = {
            'image': self.to_tensor(self.normalize(image[index]['data'])),
            'filepath': image[index]['filepath'],
            'min': torch.from_numpy(self.dataset_min),
            'max': torch.from_numpy(self.dataset_max)
        }
        
        if self.opt.recon_classification_guided:
            output['label'] = torch.from_numpy(image[index]['label']).unsqueeze(0).to(torch.float32)

        if self.opt.recon_visualize:
            output['row'] = str(image[index]['row'])
            output['col'] = str(image[index]['col'])

        return output
 

class ReconstructorValidDataset(ReconstructorDataset):

    def __init__(
        self, 
        opt: ArgumentParser,
        phase: str = 'valid'
    ) -> None:
        super().__init__(opt, phase)
        
        if phase == 'valid':
            self.dataset = [ReconstructorValidFile(opt, fp, phase) for fp in self.filepath if ('crop' not in fp) and ('synthetic' not in fp) and ('anomaly' not in fp)]
            self.get_normalization()
    
    @override
    def __getitem__(self, index: int) -> Dict[str, Tensor | str | int]:
        image_id = 0
        for image in self.dataset:
            image_id += 1
            if index < len(image):
                break
            index -= len(image)

        image = image[index]
        
        output = {
            'image': self.to_tensor(self.normalize(image['data'])),
            'filepath': image['filepath'],
            'image_id': image_id,
            'row': image['row'],
            'col': image['col']
        }

        return output


class ReconstructorPredictDataset(ReconstructorValidDataset):
    
    def __init__(
        self, 
        opt: ArgumentParser,
        phase: str = 'predict'
    ) -> None:
        super().__init__(opt, phase)
        
        filepath_dict = {
            'full': self.filepath,
            'sample_only': [s for s in self.filepath if 'crop' in s],
            'valid_only': [s for s in self.filepath if ('crop' not in s) and ('synthetic' not in s) and ('clutter' not in s) and ('mask' not in s)],
            'synthetic_only': [s for s in self.filepath if ('synthetic' in s) or ('clutter' in s)],
        }
        for k, v in filepath_dict.items():
            if self.opt.recon_data_prediction == k:
                filepath = v
                break
        
        dataset_filepaths = [fp for fp in filepath]
        self.dataset = [
            ReconstructorValidFile(opt, fp, phase) for fp in dataset_filepaths
        ]
        self.get_normalization()

        
class ReconstructorTestDataset(BaseDataset):

    def __init__(self, opt: ArgumentParser, phase: str = 'test'):
        super().__init__(opt, phase)
        
        version_number = opt.version.split('AE')[1]
        filepath_slc_all = set(glob.glob(os.path.join(self.data_dir, 'slc', '*.npy')))
        self.filepath_reconstructed = glob.glob(os.path.join(self.data_dir, f'reconstructed{version_number}', '*.npy'))
        # Find matching SLC files for each reconstructed file
        self.filepath_input = [
                fp.replace(f'reconstructed{version_number}/pred_', f'{self.data_folder}/') 
                for fp in self.filepath_reconstructed 
                if fp.replace(f'reconstructed{version_number}/pred_', 'slc/') in filepath_slc_all
            ]
        
        self.images = [
            ReconstructorTestFile(opt, filepath_slc, filepath_reconstructed) for (filepath_slc, filepath_reconstructed) in zip(self.filepath_input, self.filepath_reconstructed)
        ]

    def __len__(self) -> int:
        """Returns the total number of patches of the dataset."""
        return sum([len(image) for image in self.images])

    def __getitem__(self, index: int) -> Dict[str, Tensor | str | int]:
        """Returns the index-th patch of the associated NPY file.

        Args:
            index (int): index of the patch to be extracted

        Returns:
            Dict[str, Tensor | str | int]: Tensor and metadata
        """
        image_id = 0
        for image in self.images:
            image_id += 1
            if index < len(image):
                break
            index -= len(image)

        input, reconstructed = image[index]['input'], image[index]['reconstructed']
        return {
            'input': self.to_tensor(input),
            'reconstructed': self.to_tensor(reconstructed),
            'image_id': image_id,
            'filepath': self.filepath_input[image_id - 1],
            'row': image[index]['row'],
            'col': image[index]['col']
        }
