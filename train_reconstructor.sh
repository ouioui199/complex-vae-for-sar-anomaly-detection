#!/bin/bash                      
#SBATCH --time=00:30:00
#SBATCH --qos=co_short_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=g317,g318,g319
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --job-name=train
#SBATCH --output=train.txt

# export reconstructor workdir
export RECONSTRUCTOR_WORKDIR=... # specify the reconstructor workdir here

cleanup() {
    # Add your cleanup code here
    echo "Script interrupted, cleaning up..."
    # Kill any background processes
    jobs -p | xargs -r kill
    exit 1
}

# Set the trap
trap cleanup SIGINT SIGTERM

VERSION_NUMBER=... # specify the version number here
VERSION="test_complexVAE${VERSION_NUMBER}"
DATADIR=... # specify the data directory here
DATABAND=... # specify the data band here
LOGDIR=training_logs
PATCH_SIZE=64
STRIDE=... # specify the stride here
EPOCHS=... # specify the number of epochs here
LATENT_COMPRESSION=... # specify the latent compression here
LR=... # specify the learning rate here
TRAIN_BATCH_SIZE=... # specify the training batch size here
VAL_BATCH_SIZE=... # specify the validation batch size here
IN_CHANNELS=... # specify the input channels here
KLD_WEIGHT=... # specify the KLD weight here
BETA_WARMUP=... # specify the beta warmup epochs here
BETA_N_EPOCHS=... # specify the beta n epochs here

python scripts/train_reconstructor.py \
    --version $VERSION \
    --data_band $DATABAND \
    --datadir $DATADIR \
    --recon_patch_size $PATCH_SIZE \
    --recon_stride $STRIDE \
    --recon_in_channels $IN_CHANNELS \
    --recon_epochs $EPOCHS \
    --recon_latent_compression $LATENT_COMPRESSION \
    --recon_lr_ae $LR \
    --recon_train_slc \
    --kld_weight $KLD_WEIGHT \
    --recon_beta_warmup_epochs $BETA_WARMUP \
    --recon_train_batch_size $TRAIN_BATCH_SIZE \
    --recon_val_batch_size $VAL_BATCH_SIZE \
    --beta_n_epochs $BETA_N_EPOCHS \
    --recon_regulate_beta

echo "Done ${VERSION}"
