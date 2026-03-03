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

EXCLUSION_WINDOW_SIZE=... # set this to your desired exclusion window size
BOX_CAR_SIZE=... # set this to your desired box car size
DATA_DIR=... # set this to your data directory
RX_TYPE=... # set this to your desired RX type
RX_DATA='slc' # set this to your desired RX data type (e.g., 'slc' or 'pauli')
RX_CHUNK_BATCH_SIZE=... # set this to your desired RX chunk batch size so that it can fit in GPU memory and optimize speed

mkdir -p $SAVE_DIR

python compute_RX.py \
    --version 0 \
    --data_band L \
    --rx_exclusion_window_size $EXCLUSION_WINDOW_SIZE \
    --rx_box_car_size $BOX_CAR_SIZE \
    --datadir $DATA_DIR \
    --rx_type $RX_TYPE \
    --rx_data $RX_DATA \
    --rx_chunk_batch_size $RX_CHUNK_BATCH_SIZE \
    --undersample_dso # Comment this if you don't want to undersample the DSO for RX computation. 
    # --rx_real_valued # Uncomment this if you want to compute RX on real-valued data (e.g., amplitude or intensity) instead of complex-valued data (e.g., SLC).

echo "Done "
