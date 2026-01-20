from argparse import Namespace
from typing import Sequence, Dict
from pathlib import Path
import os

import torch
import numpy as np
from torch import nn, Tensor
import torchcvnn.nn as c_nn
from torch.optim import Adam
from torchcvnn.nn.modules.loss import ComplexMSELoss

from models.networks import complexVAE
from models.loss import complex_kullback_leibler_divergence_loss
from models.utils import check_path
from models.base_model import BaseModel


class ReconstructorModule(BaseModel):

    def __init__(self, opt: Namespace, image_out_dir: str) -> None:
        super().__init__(opt, image_out_dir)
        
        self.train_step_outputs = {}
    
    def on_fit_start(self) -> None:
        check_path(self.image_save_dir)

    def on_train_epoch_end(self) -> None:
        tb_logger = self.loggers[0].experiment
        _log_dict = {
            key.replace('step_', ''): torch.tensor(value).mean()
            for (key, value) in self.train_step_outputs.items()
        }

        _log_dict_loss = {f'Loss/{key}': value for (key, value) in _log_dict.items() if 'loss' in key}
        for k,v in _log_dict_loss.items():
            tb_logger.add_scalar(k, v, self.current_epoch)
            
        self.train_step_outputs.clear()

        return tb_logger, _log_dict

    def _on_val_pred_start(self, dataset: torch.utils.data.Dataset) -> None:
        self.pred = []
        self.patch_overlap_count = []
        
        for _, data in enumerate(dataset.dataset):
            shape = data.data.shape
            dtype = torch.from_numpy(np.array([0], dtype=data.data.dtype)).dtype
            self.pred.append(torch.zeros(shape, dtype=dtype, device=self.device))
            self.patch_overlap_count.append(torch.zeros(shape, dtype=dtype, device=self.device))

    def on_validation_epoch_start(self) -> None:
        dataset = self.trainer.datamodule.valid_dataset
        self._on_val_pred_start(dataset)
        self.val_reconstruction_loss = []

    def _validation_step(self, pred: Tensor, image: Tensor, batch: Dict[str, Tensor | str], batch_idx: int) -> None:
        batch_size = len(image)
        patch_size = self.opt.recon_patch_size

        image_ids = batch['image_id'] - 1
        cols = batch['col']
        rows = batch['row']
        
        for i in range(batch_size):
            img_id = image_ids[i]
            col_start, row_start = cols[i], rows[i]
            col_end, row_end = col_start + patch_size, row_start + patch_size

            self.pred[img_id][:, col_start : col_end, row_start : row_end].add_(pred[i])
            self.patch_overlap_count[img_id][:, col_start : col_end, row_start : row_end].add_(torch.ones_like(pred[i]))

    def on_validation_epoch_end(self) -> None:
        tb_logger = self.loggers[1].experiment
        
        val_reconstruction_loss_ = torch.tensor(self.val_reconstruction_loss).mean()
        self.log_dict({'val_rec_loss': val_reconstruction_loss_}, prog_bar=True)
        tb_logger.add_scalar('Loss/rec_loss', val_reconstruction_loss_, self.current_epoch)

        del self.pred, self.patch_overlap_count, self.val_reconstruction_loss
        torch.cuda.empty_cache()

    def on_predict_start(self) -> None:
        os.makedirs(self.image_save_dir, exist_ok=True)
        
        dataset = self.trainer.datamodule.pred_dataset
        self._on_val_pred_start(dataset)

    def predict_step(self, batch: Dict[str, Tensor | str], batch_idx: int) -> None:
        self.validation_step(batch, batch_idx, compute_loss=False)

    def on_predict_end(self) -> None:
        dataset = self.trainer.datamodule.pred_dataset
        
        for i, data in enumerate(dataset.dataset):
            image_ids = i - 1
            min, max = dataset.dataset_min, dataset.dataset_max
            denorm = self.denorm(min, max)
            
            pred_ = torch.div(self.pred[image_ids], self.patch_overlap_count[image_ids])
            pred_ = pred_.cpu().data.numpy()
            pred_ = denorm(pred_)
            
            name, ext = self.get_name_ext(data.filepath, add_epoch=False)
            np.save(Path(f'{self.image_save_dir}/pred_{name}{ext}'), pred_)

    def on_test_start(self) -> None:
        version_number = self.opt.version.split("AE")[1]
        self.anomaly_map_dir = str(self.image_save_dir).replace(f'reconstructed{version_number}', f'rec_cplxVAE{version_number}_full_slc_{self.opt.recon_anomaly_kernel}')
        os.makedirs(self.anomaly_map_dir, exist_ok=True)
        
        dataset = self.trainer.datamodule.test_dataset.images
        self.anomaly_map = []
        for i, image in enumerate(dataset):
            dtype = torch.from_numpy(np.array([0], dtype=image.input.dtype)).dtype
            self.anomaly_map.append(torch.zeros((image.height, image.width), dtype=dtype, device=self.device))

    def test_step(self, batch: Dict[str, Tensor | str], batch_idx: int) -> None:
        pred, input = batch['reconstructed'], batch['input']
        
        pred_cov, _ = self.compute_scm_smv(pred.view(*pred.shape[:2], -1))
        input_cov, _ = self.compute_scm_smv(input.view(*input.shape[:2], -1))
        
        anomaly_score = torch.linalg.norm(pred_cov - input_cov, ord='fro', dim=(1, 2)) ** 2
        
        image_ids = batch['image_id'] - 1
        cols = batch['col']
        rows = batch['row']
        for (id, col, row, score) in zip(image_ids, cols, rows, anomaly_score):
            self.anomaly_map[id][col, row] = score

    def on_test_end(self) -> None:
        dataset = self.trainer.datamodule.test_dataset
        
        for i, data in enumerate(dataset.images):
            input = data.input
            pred = data.reconstructed
            
            frobenius_anomaly_map = self.anomaly_map[i].real.cpu().data.numpy()
            manhattan_anomaly_map, sum_manhattan_anomaly_map = self.create_manhattan_anomaly_map(pred, input)
            euclidean_anomaly_map, sum_euclidean_anomaly_map = self.create_euclidean_anomaly_map(pred, input)
            
            name, ext = self.get_name_ext(dataset.filepath_input[i], add_epoch=False) # Attention here, check if i is really the image
            np.save(Path(f'{self.anomaly_map_dir}/frobenius_anomaly_{name}{ext}'), frobenius_anomaly_map)
            np.save(Path(f'{self.anomaly_map_dir}/manhattan_anomaly_{name}{ext}'), manhattan_anomaly_map)
            np.save(Path(f'{self.anomaly_map_dir}/sum_manhattan_anomaly_{name}{ext}'), sum_manhattan_anomaly_map)
            np.save(Path(f'{self.anomaly_map_dir}/euclidean_anomaly_{name}{ext}'), euclidean_anomaly_map)
            np.save(Path(f'{self.anomaly_map_dir}/sum_euclidean_anomaly_{name}{ext}'), sum_euclidean_anomaly_map)
            np.save(Path(f'{self.anomaly_map_dir}/diff_anomaly_{name}{ext}'), pred - input)


class VAEModule(ReconstructorModule):

    def __init__(self, opt: Namespace, image_out_dir: str) -> None:
        super().__init__(opt, image_out_dir)

    def forward(self, x: Tensor) -> Sequence[Tensor]:
        if self.opt.recon_model == 'cplxvae':
            z, mu, sigma, delta = self.model(x)
            return self.model.decode(z), mu, sigma, delta
        
        z, mu, log_var = self.model(x)
        return self.model.decode(z), mu, log_var
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return Adam(self.model.parameters(), lr=self.opt.recon_lr_ae)
    
    def get_current_beta(self) -> float:
        if self.opt.recon_beta_proportion > 1 or self.opt.recon_beta_proportion < 0:
            raise ValueError("R must be between 0 and 1")
        
        # Warmup
        if self.current_epoch < self.opt.recon_beta_warmup_epochs:
            return self.opt.recon_beta_start
        
        if not self.opt.recon_regulate_beta:
            return self.opt.recon_beta_end
        
        current_beta_epoch = self.current_epoch + 1 - self.opt.recon_beta_warmup_epochs
        if self.opt.beta_warmup_type == 'linear' and current_beta_epoch <= self.total_beta_iterations:
        #Linear annealing
            return min(
                self.opt.recon_beta_end, 
                self.opt.recon_beta_start + (self.opt.recon_beta_end - self.opt.recon_beta_start) * current_beta_epoch / self.total_beta_iterations
            )
        elif self.opt.beta_warmup_type == 'cyclical':
            # Cyclical annealing
            cycle_size = self.total_beta_iterations // 4
            beta_step = self.global_step - self.opt.recon_beta_warmup_epochs * self.trainer.num_training_batches
            tau = (beta_step % cycle_size) / cycle_size
            if tau < self.opt.recon_beta_proportion:
                return self.opt.recon_beta_start + (
                    self.opt.recon_beta_end - self.opt.recon_beta_start
                ) * (tau / self.opt.recon_beta_proportion)

        return self.opt.recon_beta_end

    def on_train_start(self) -> None:
        if self.trainer.max_epochs < 20:
            raise ValueError("Training epochs should be at least 20")
        self.max_beta_epochs = self.trainer.max_epochs - 5
        if self.opt.beta_n_epochs is None:
            self.total_beta_iterations = (self.max_beta_epochs - self.opt.recon_beta_warmup_epochs) * self.trainer.num_training_batches
        else:
            self.total_beta_iterations = self.opt.beta_n_epochs

    def _training_step(self, reconstruction: Tensor, image: Tensor, reconstruction_loss: Tensor, kld_loss: Tensor, classification_loss: Tensor) -> Tensor:
        beta = self.get_current_beta()
        loss = reconstruction_loss + beta * kld_loss + classification_loss

        if self.global_step % (self.trainer.num_training_batches * 25) == 0:
            reconstruction = torch.abs(reconstruction[0])
            image = torch.abs(image)
            
            self.log_image("Input", (image * 255).to(torch.uint8))
            self.log_image("Reconstruction", (reconstruction * 255).to(torch.uint8))
            self.log_image("Difference", ((reconstruction - image) * 255).to(torch.uint8))

        if not self.train_step_outputs:
            self.train_step_outputs = {
                "step_loss": [loss.detach()],
                "step_rec_loss": [reconstruction_loss.detach()],
                "step_kld_loss": [kld_loss.detach()],
                "step_classification_loss": [classification_loss.detach()],
                "step_beta": [beta]
            }
        else:
            self.train_step_outputs["step_loss"].append(loss.detach())
            self.train_step_outputs["step_rec_loss"].append(reconstruction_loss.detach())
            self.train_step_outputs["step_kld_loss"].append(kld_loss.detach())
            self.train_step_outputs["step_classification_loss"].append(classification_loss.detach())
            self.train_step_outputs["step_beta"].append(beta)
            
        return loss
    
    def validation_step(self, batch: Dict[str, Tensor | str], batch_idx: int, compute_loss: bool) -> None:
        image = batch['image']
        pred_ = self(image)[0]

        if compute_loss:
            self.val_reconstruction_loss.append(self.reconstruction_loss(pred_, image))
            
        super()._validation_step(pred_, image, batch, batch_idx)

    def on_train_epoch_end(self) -> None:
        tb_logger, _log_dict = super().on_train_epoch_end()
        tb_logger.add_scalar('Beta', _log_dict['beta'], self.current_epoch)


class ComplexVAEModule(VAEModule):
    
    def __init__(self, opt: Namespace, image_out_dir: str) -> None:
        super().__init__(opt, image_out_dir)

        self.model = complexVAE(
            num_channels=self.opt.recon_in_channels,
            num_layers=4,
            channels_ratio=32,
            activation=c_nn.modReLU(),
            latent_compression=self.opt.recon_latent_compression
        )
        self.model.initialize_weights()
        self.kld_loss = complex_kullback_leibler_divergence_loss
        self.reconstruction_loss = ComplexMSELoss()
        self.classification_loss = nn.BCEWithLogitsLoss()

    def training_step(self, batch: Dict[str, str | Tensor], batch_idx: int) -> Tensor:
        image = batch['image']

        reconstruction, mu, sigma, delta = self(image)
        reconstruction_loss = self.reconstruction_loss(reconstruction[0], image)
        kld_loss = self.kld_loss(mu, sigma, delta, self.opt.kld_weight)
        classification_loss = torch.tensor(0., device=self.device)
        
        if self.opt.recon_classification_guided:
            classification_loss = self.classification_loss(reconstruction[1], batch['label'])
            
        return super()._training_step(reconstruction, image, reconstruction_loss, kld_loss, classification_loss)
    
    def validation_step(self, batch: Dict[str, str | Tensor], batch_idx: int, compute_loss: bool = True) -> None:
        return super().validation_step(batch, batch_idx, compute_loss)
