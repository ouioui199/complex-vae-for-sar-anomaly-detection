import os, glob
from argparse import ArgumentParser
from typing import Callable
from pathlib import Path

import torch
from lightning import Trainer

import utils as U
from models import M_M


def reconstructor_predict(opt: ArgumentParser, trainer: Callable, workdir: str) -> None:
    ckpt_path = glob.glob(str(Path(workdir)/f'weights_storage/version_{opt.version}/reconstructor/*.ckpt'))
    ckpt_path = next((p for p in ckpt_path if (f"{opt.data_band}-band" in p) and ('best' in p)), None)
    if ckpt_path:
        # Define data module
        data_module = U.ReconstructionDatasetModule(opt)
        data_module.setup(stage='predict')
        image_out_dir = data_module.pred_dataset.data_dir
        image_out_dir = image_out_dir.replace('slc', f'reconstructed{opt.version.split("AE")[1]}')
        # Define reconstruction module
        model = M_M.ComplexVAEModule.load_from_checkpoint(ckpt_path, opt=opt, image_out_dir=image_out_dir)
        # Predict
        if opt.recon_predict:
            trainer.predict(model, datamodule=data_module)
        # Test
        data_module.setup(stage='test')
        trainer.test(model, datamodule=data_module)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = U.ArgumentParsing(parser)
    parser.predict_reconstructor_args(parser.predict_reconstructor_group)
    opt = parser.parser.parse_args()

    if opt.recon_anomaly_kernel % 2 == 0:
        raise ValueError("Anomaly kernel size must be an odd number")
    
    torch.backends.cudnn.enabled = True
    torch.set_float32_matmul_precision('high')
    
    workdir = os.getenv('RECONSTRUCTOR_WORKDIR', '')
    reconstructor = Trainer(
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        benchmark=True,
        logger=False,
        enable_progress_bar=False
        # callbacks=U.CustomProgressBar()
    )
    reconstructor_predict(opt, reconstructor, workdir)
