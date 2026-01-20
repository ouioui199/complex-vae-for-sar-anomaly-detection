import glob, sys
from typing import Dict
from pathlib import Path
from argparse import ArgumentParser

import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchcvnn.transforms import ToTensor
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

sys.path.append('./scripts')
sys.path.append('./scripts/datasets')
sys.path.append('./scripts/models')
from scripts.datasets.transforms import MinMaxNormalize
from scripts.models import M_M
from scripts.utils import ArgumentParsing


class TSNEFile:

    class_map = {
        'anomalies': 1,
        'crops': 2,
        'land1': 3,
        'land2': 4,
        'road': 5,
        'tree': 6
    }

    def __init__(
        self, 
        opt: ArgumentParser,
        filepath: str
    ) -> None:
        """Return the whole validation part of the image"""
        self.opt = opt
        
        self.input = np.load(filepath, mmap_mode='r')
        self.input = np.abs(self.input) if 'cplx' not in opt.recon_model else self.input
        self.data_class = filepath.split('/')[-2]

        _, nrows, ncols = self.input.shape
        self.nsamples_per_rows = (ncols - opt.tsne_patch_size) // opt.tsne_stride + 1
        self.nsamples_per_cols = (nrows - opt.tsne_patch_size) // opt.tsne_stride + 1

    def __len__(self) -> int:
        return self.nsamples_per_rows * self.nsamples_per_cols
    
    def __getitem__(self, index: int) -> Dict[str, Tensor | str | int]:
        row = index // self.nsamples_per_rows
        col = index % self.nsamples_per_rows

        row_start = row * self.opt.tsne_stride
        col_start = col * self.opt.tsne_stride

        return {
            'data': self.input[:, row_start : row_start + self.opt.tsne_patch_size, col_start : col_start + self.opt.tsne_patch_size],
            'class': self.class_map[self.data_class]
        }
    

class TSNEDataset(Dataset):

    def __init__(
            self, 
            opt: ArgumentParser
    ) -> None:

        self.dataset = [TSNEFile(opt, fp) for fp in glob.glob(f'{opt.datadir}/{opt.data_band}_band/classes/*/*.npy')]
        self.normalize = MinMaxNormalize(opt.tsne_dataset_min, opt.tsne_dataset_max)
        self.to_tensor = ToTensor('float32') if opt.recon_model != 'cplxvae' else ToTensor('complex64')
    
    def __len__(self) -> int:
        return sum([len(image) for image in self.dataset])

    def __getitem__(self, index: int) -> Dict[str, Tensor | str | int]:
        for image in self.dataset:
            if index < len(image):
                break
            index -= len(image)
        
        output = {
            'image': self.to_tensor(self.normalize(image[index]['data'])),
            'class': image[index]['class']
        }

        return output


class computeTSNE:

    def __init__(self, opt: ArgumentParser) -> None:
        self.opt = opt

        dataset = TSNEDataset(opt)
        self.dataloader = DataLoader(
            dataset, 
            batch_size=opt.tsne_batch_size, 
            shuffle=False, 
            num_workers=opt.workers
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt_path = glob.glob(str(Path(f'{opt.workdir}/projects/reconstructor/weights_storage/version_{opt.version}/reconstructor/*.ckpt')))
        ckpt_path = next((p for p in ckpt_path if (f"{opt.data_band}-band" in p) and ('best' in p)), None)
        self.model = M_M.ComplexVAEModule.load_from_checkpoint(ckpt_path, opt=opt, image_out_dir=None)

    def forward(self):
        self.model.eval()
        latent_vectors, classes = [], []
        with torch.no_grad():
            for i, data in enumerate(self.dataloader):
                classes.extend(data['class'])

                data = data['image'].to(self.device)
                z = self.model.model(data)[0] if 'vae' in self.opt.recon_model else self.model(data)
                latent_vectors.append(z)

            latent_vectors = torch.cat(latent_vectors, dim=0).cpu().numpy()
            classes = np.array(classes)

        return latent_vectors, classes
    
    def compute_tsne(self):
        # latent_pca, classes = self.compute_pca()
        latent_vectors, classes = self.forward()
        tsne = TSNE(n_components=2, random_state=47, perplexity=50)
        latent_tsne = tsne.fit_transform(np.abs(latent_vectors.reshape(latent_vectors.shape[0], -1)))

        return latent_tsne, classes
    
    def plot_tsne(self):
        latent_tsne, classes = self.compute_tsne()
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=classes, cmap='tab20', alpha=0.6)
        plt.legend(*scatter.legend_elements(), title="Classes")
        plt.title('t-SNE of Latent Space')
        plt.savefig(f'tsne_plot_version_{self.opt.version}.png')
        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = ArgumentParsing(parser)
    parser.compute_tsne_args(parser.compute_tsne_group)
    opt = parser.parser.parse_args()

    computeTSNE(opt).plot_tsne()
