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
#SBATCH --job-name=pred_despeckler
#SBATCH --output=pred_despeckler.txt

# export reconstructor workdir
export RECONSTRUCTOR_WORKDIR=/mnt/DATA/projects/reconstructor

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
DATABAND=... # specify the data band here
DATADIR=... # specify the data directory here
PATCH_SIZE=64
ANOMALY_KERNEL=9
LATENT_COMPRESSION=... # specify the latent compression here
STRIDE=... # specify the stride here
PREDICTION_DATA_TYPE=... # specify the prediction data type here
WORKERS=4
PRED_BATCH_SIZE=... # specify the prediction batch size here
TEST_BATCH_SIZE=... # specify the test batch size here
NORM_VALUES=... # specify the normalization values here

python scripts/predict_reconstructor.py \
    --version $VERSION \
    --data_band $DATABAND \
    --datadir $DATADIR \
    --recon_latent_compression $LATENT_COMPRESSION \
    --recon_patch_size $PATCH_SIZE \
    --recon_anomaly_kernel $ANOMALY_KERNEL \
    --recon_stride $STRIDE \
    --recon_data_prediction $PREDICTION_DATA_TYPE \
    --recon_train_slc \
    --workers $WORKERS \
    --recon_pred_batch_size $PRED_BATCH_SIZE \
    --recon_test_batch_size $TEST_BATCH_SIZE \
    --normalization_values "$NORM_VALUES" \
    --recon_predict

echo "Done cplxVAE${VERSION_NUMBER} ${ANOMALY_KERNEL}x${ANOMALY_KERNEL}!"

ANOMALY_KERNEL=5

python scripts/predict_reconstructor.py \
    --version $VERSION \
    --data_band $DATABAND \
    --datadir $DATADIR \
    --recon_latent_compression $LATENT_COMPRESSION \
    --recon_patch_size $PATCH_SIZE \
    --recon_anomaly_kernel $ANOMALY_KERNEL \
    --recon_stride $STRIDE \
    --recon_data_prediction $PREDICTION_DATA_TYPE \
    --recon_train_slc \
    --workers $WORKERS \
    --recon_pred_batch_size $PRED_BATCH_SIZE \
    --recon_test_batch_size $TEST_BATCH_SIZE \
    --normalization_values "$NORM_VALUES"

echo "Done cplxVAE${VERSION_NUMBER} ${ANOMALY_KERNEL}x${ANOMALY_KERNEL}!"
