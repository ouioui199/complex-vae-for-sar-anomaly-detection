# Anomaly Detection in SAR imaging

## Abstract
We propose an unsupervised learning approach for anomaly detection in SAR imaging. The proposed model combines a preprocessing despeckling step, a $\beta$ Variational Auto-Encoder (VAE) for unsupervised anomaly filtering, and an anomaly detector based on the change of the covariance matrix at the input and output of the $\beta$-VAE network. Experiments have been carried out on X-band ONERA polarimetric SAR images to demonstrate the effectiveness of Beta VAE compared with the methods proposed in the literature.

## Architecture
Real-valued Variational AutoEncoder
![VAE architecture](images/VAE-6c.png)

Complex-valued Variational AutoEncoder
![VAE architecture](images/complexVAE.png)

## Getting started
Anomaly Detection in SAR imaging with Adversarial AutoEncoder, Variational AutoEncoder and Reed-Xiaoli Detector.
To begin, clone the repository with ssh or https:

```
git clone git@gitlab-research.centralesupelec.fr:anomaly-detection-huy/aae_huy.git
git clone https://gitlab-research.centralesupelec.fr/anomaly-detection-huy/aae_huy.git
```

### Environment
Create a virtual environment with miniconda or other tools.
Details to install miniconda could be found [here](https://www.anaconda.com/docs/getting-started/miniconda/install).

### Install requirements
```
pip install -r requirements.txt
```

Install torchcvnn latest developments and install it as a library
```
git clone --single-branch --branch dev_transforms https://github.com/ouioui199/torchcvnn.git
pip install -e torchcvnn
```

We will use Pytorch-Lightning to organize our code. Documentations can be found [here](https://lightning.ai/docs/pytorch/stable/starter/introduction.html)

## Data folder structure
For quad-polarization images, the data folder container MUST be organized like below. Create folders if needed.
```
|- data_folder1/
|   |- L_band/
|   |- UHF_band/
|   |- X_band/
|   |   |- train/
|   |   |- predict/
|   |   |   |- despeckled/
|   |   |   |- reconstructed/
|   |   |   |- slc/
|   |   |   |   |- something_Combined_something.npy
|   |   |   |   |- something_Hh_something.npy
|   |   |   |   |- something_Hv_something.npy
|   |   |   |   |- something_Vh_something.npy
|   |   |   |   |- something_Vv_something.npy
|- data_folder2/
|   |- L_band/
|   |   |- train/
|   |   |   |- despeckled/
|   |   |   |- reconstructed/
|   |   |   |- slc/
|   |   |   |   |- something_Combined_something.npy
|   |   |   |   |- something_Hh_something.npy
|   |   |   |   |- something_Hv_something.npy
|   |   |   |   |- something_Vh_something.npy
|   |   |   |   |- something_Vv_something.npy
|   |   |- predict/
|   |- UHF_band/
etc.
```
HvVh polarization should be pre-computed from Hv and Vh, with HvVh = (Hv + Vh) / 2. Normalization values will be computed automatically during the data processing, you don't need to do anything. **If the 'something_Combined_something.npy' files aren't available, run the 'combine_polar_channels' function in scripts/utils.py file.**

## Data preparation
Let's say you wish to train with data stored in ```DATADIR=/your/data/folder/data_folder1```. The code will operate as below:

![Workflow](images/Workflow.png)

**Note: Preparing despeckled data are not mandatory if you wish to train directly on Single Look Complex.**

Anomaly map computed with the Reed-Xiaoli detector can be computed with the command below. Note that this code works with 4 polarization SAR images.
```python
python compute_RX.py --version 0 --data_band your-choice --datadir /your/data/folder/data_folder1 --rx_box_car_size your-choice --rx_exclusion_window_size your-choice
```

## Training

### Step 1
To work with despeckled data, first, you need to train the despeckler. We will use [MERLIN](https://ieeexplore.ieee.org/document/9617648) algorithms. Otherwise, skip to step 2.

The code will outputs and save checkpoints to ```weights_storage/version_X/despeckler/*.ckpt```. Remember to change 'X' to your version. In the shell file has already been programmed to run sequentially 4 channels of a full polarization SAR image. If you wish to run it only on certain channel, comment the concerned code.
```
bash train_despeckler.sh > train_despeckler_log.txt 2>&1
```

The despeckler training has now finished, you must compute predictions to get despeckled SAR images. To compute predictions, run
```
bash predict_despeckler.sh > pred_despeckler_log.txt 2>&1
```

### Step 2

To train the reconstruction network (VAE, complex-valued VAE or AAE), run
```
bash train_reconstructor.sh > train_recon_log.txt 2>&1
```

To train or predict with Single Look Complex images, remove the argument ```--recon_train_slc``` in the shell script.

To perform predictions, run
```
bash predict_reconstructor.sh > pred_recon_log.txt 2>&1
```

## Folder structure
Once start running the code, the folder will be organized as below. After cloning the code, create environments, install dependencies, you can start immediately the training. No further actions are required. All folders will be created automatically.
```
|- images/
|- scripts/
|   |- datasets/
|   |- models/
|   |- predict_despeckler.py
|   |- predict_reconstructor.py
|   |- train_despeckler.py
|   |- train_reconstructor.py
|   |- utils.py
|- training_logs/
|   |- version X/
|   |   |- despeckler
|   |   |   |- visualization
|   |   |   |- validation_samples
|   |   |- reconstructor
|   |   |   |- visualization
|   |   |   |- validation_samples
|- weights_storage/
|   |- version X/
|   |   |- despeckler
|   |   |- reconstructor
|- .gitignore
|- compute_RX.py
|- compute_tsne.py
|- create_synthetic_anomalies.py
|- README.md
|- requirements.txt
|- predict_despeckler.sh
|- predict_reconstructor.sh
|- train_despeckler.sh
|- train_reconstructor.sh
```
