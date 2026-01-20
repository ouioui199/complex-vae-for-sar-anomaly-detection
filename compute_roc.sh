#!/bin/bash                      
#SBATCH --time=12:00:00
#SBATCH --qos=co_inter_std
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=22
#SBATCH --mem=128G
#SBATCH --job-name=roc
#SBATCH --output=roc.txt

# export reconstructor workdir
export RECONSTRUCTOR_WORKDIR=... # set this to your reconstructor workdir

# Train despeckler ubuntu pc script
cleanup() {
    # Add your cleanup code here
    echo "Script interrupted, cleaning up..."
    # Kill any background processes
    jobs -p | xargs -r kill
    exit 1
}

# Set the trap
trap cleanup SIGINT SIGTERM

VERSION=... # set this to the version number of your trained model
BETA=... # set this to the beta value used in your trained model
LATENT_COMPRESSION=...... # set this to the latent compression value used in your trained model
FOLDER_NAME="cVAE_beta${BETA}_lc${LATENT_COMPRESSION}"
SAVE_DIR="$RECONSTRUCTOR_WORKDIR/projects/reconstructor/roc_curves/${FOLDER_NAME}"

mkdir -p $SAVE_DIR

python compute_roc.py \
    --path_label "$RECONSTRUCTOR_WORKDIR/data/AnomalyDetection/dessous/X_band/predict/slc/anomaly_mask_20210705_151308_01b_X45-Combined-strip_IA_rad_com1.npy" \
    --save_dir $SAVE_DIR \
    --anomaly_map_dirs "$RECONSTRUCTOR_WORKDIR/data/AnomalyDetection/dessous/X_band/train/slc/" "$RECONSTRUCTOR_WORKDIR/data/AnomalyDetection/dessous/X_band/predict/rec_cplxVAE${VERSION}_full_slc_9/" "$RECONSTRUCTOR_WORKDIR/data/AnomalyDetection/dessous/X_band/predict/rec_cplxVAE${VERSION}_full_slc_5/"

echo "Done${VERSION}"
