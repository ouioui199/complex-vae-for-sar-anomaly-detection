from pathlib import Path
from typing import Sequence
from argparse import ArgumentParser, Namespace
from collections import defaultdict
import glob, os

import numpy as np
import jax.numpy as jnp
from jax import Array, vmap, device_get
from sklearn.metrics import auc
from matplotlib import pyplot as plt
from tqdm import tqdm


def match_shape_image_and_label(
    anomaly_map: Array,
    label: Array,
    box_car_size: int, 
    crop: bool, rx: bool
):
    if len(anomaly_map.shape) == 3 and anomaly_map.shape[0] != 1:
        raise ValueError('Anomaly map should be a 2D array or a 3D array with a single channel.')
    # Compute slice indices once (unified for both tensors)
    end_offset = 0 if rx else 1
    slice_indices = (
        slice(box_car_size // 2, -box_car_size // 2 + end_offset), 
        slice(box_car_size // 2, -box_car_size // 2 + end_offset)
    )
    # Apply slicing to both tensors
    label = label[slice_indices]
    anomaly_map = anomaly_map if crop else anomaly_map[slice_indices]
    anomaly_map = anomaly_map[0] if rx else anomaly_map
    
    return anomaly_map, label


def get_anomaly_map_and_label_synthetic_anomalies(
    opt: Namespace, 
    path_anomaly_map: str, 
    path_clutter: str, 
    box_car_size: int, 
    crop: bool, rx: bool
) -> Sequence[Array]:
    # Load label
    label = jnp.load(opt.path_label, mmap_mode='r')
    # Load and preprocess anomaly map
    anomaly_map = jnp.load(str(Path(path_anomaly_map)))
    if path_clutter != None:
        # Load clutter for thresholding
        clutter = jnp.load(path_clutter, mmap_mode='r').flatten()
    else:
        clutter = None
    # Match shapes of anomaly map and label, and apply cropping if needed
    anomaly_map, label = match_shape_image_and_label(anomaly_map, label, box_car_size, crop, rx)
    return clutter, anomaly_map, label


def compute_roc_labeled_anomalies(anomaly_map: Array, label: Array):
    # Sort clutter to get thresholds
    n = anomaly_map.flatten().shape[0]
    # Total positive samples
    label_pos = label == 1
    total_pos = jnp.sum(label_pos)
    # Probability of False Alarm values corresponding to the thresholds
    pfa = jnp.arange(1, n + 1) / n
    tp = jnp.zeros_like(pfa)
    # Compute thresholds based on percentiles of the anomaly map
    thresholds = jnp.percentile(anomaly_map, 100 - pfa * 100)
    # Helper function to compute true positives for a given threshold
    for i, threshold_value in enumerate(thresholds):
        ano_pfa = jnp.clip(anomaly_map, threshold_value, anomaly_map.max())
        ano_pfa = (ano_pfa - ano_pfa.min()) / (ano_pfa.max() - ano_pfa.min())
        ano_pfa = ano_pfa > 0
        tp = tp.at[i].set(jnp.sum(ano_pfa & label_pos))
    # Probability of Detection and Probability of False Alarm
    pd = tp / total_pos
    return (
        np.asarray(pd),
        np.asarray(pfa),
        np.asarray(thresholds),
    )


def compute_roc_synthetic_anomalies(anomaly_map: Array, clutter: Array, label: Array) -> Sequence[np.ndarray]:
    # Sort clutter to get thresholds
    clutter = jnp.sort(clutter)
    thresholds = clutter[::-1]
    n = thresholds.shape[0]
    # Total positive samples
    label_pos = label == 1
    total_pos = jnp.sum(label_pos)
    # Helper function to compute true positives for a given threshold
    def tp_for_threshold(threshold_value: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum((anomaly_map >= threshold_value) & label_pos)
    # Vectorized computation of true positives across all thresholds
    tp = vmap(tp_for_threshold)(thresholds)
    # Probability of Detection and Probability of False Alarm
    pd = tp / total_pos
    pfa = jnp.arange(1, n + 1) / n
    # Return as numpy arrays
    return (
        np.asarray(device_get(pd)),
        np.asarray(device_get(pfa)),
        np.asarray(device_get(thresholds)),
    )


def plot_graphs_labeled_anomalies(opt: Namespace) -> None:
    # Create subplots
    _, ax_roc = plt.subplots()
    _, ax_roc_log_pfa = plt.subplots()
    _, ax_pfa_thresh_rx = plt.subplots()
    _, ax_log_pfa_thresh_rx = plt.subplots()
    _, ax_pfa_thresh_cvae = plt.subplots()
    _, ax_log_pfa_thresh_cvae = plt.subplots()
    # Define colors and linestyles
    colors = ['blue', 'red', 'green', 'cyan', 'orange']
    linestyles = ['-', '--', ':', '-.', '-']
    # Check if enough colors are defined
    if len(opt.anomaly_map_dirs) > len(colors):
        raise ValueError('Not enough colors defined for the number of anomaly map directories provided.')
    # First filtering: only RX-SCM, RX-Tyler, Frobenius; no real_valued; undersampled; correct boxcar size; no guard window
    paths_anomaly_map = [
        path for anomaly_map_dir in opt.anomaly_map_dirs 
        for path in glob.glob(f'{anomaly_map_dir}/*.npy')
        if ('RX-SCM' in path or 'frobenius' in path or 'RX-Tyler' in path) and ('real_valued' not in path) and ('undersampled' in path) and (f'_{opt.dso_boxcar}' in path) and ('no_guard_window' not in path)
    ]
    # Iterate over each anomaly map path and compute ROC curves
    for i, pam in enumerate(paths_anomaly_map):
        # Determine if cropping is needed
        crop = any(sub in Path(pam).stem for sub in ['RX', 'frobenius'])
        # Determine if RX method
        rx = ('RX-SCM' in Path(pam).stem) or ('RX-Tyler' in Path(pam).stem)
        # Determine label name. ComplexVAE or Reed-Xiaoli
        label_name = 'complexVAE' if any(sub in Path(pam).stem for sub in ['frobenius', 'manhattan', 'euclidean']) else 'RX'
        # RX-Tyler or RX-SCM
        label_name = label_name + '-Tyler' if 'Tyler' in Path(pam).stem else label_name + '-SCM' if 'RX-SCM' in Path(pam).stem else label_name
        # Load data
        _, anomaly_map, label = get_anomaly_map_and_label_synthetic_anomalies(opt, pam, None, opt.dso_boxcar, crop, rx)
        # Compute ROC
        pd, pfa, thresholds = compute_roc_labeled_anomalies(anomaly_map, label)
        # Append boxcar size if applicable
        label_name = label_name + f' - Boxcar {opt.dso_boxcar}'
        # Convert pfa to log10 scale
        log10_pfa = np.log10(pfa + 1e-10)
        # Plotting
        ax_roc.plot(
            pfa, pd, 
            label=f'{label_name} - AUC: {auc(pfa, pd):.3f}',
            color=colors[i], 
            linestyle=linestyles[i], 
            linewidth=1
        )
        ax_roc_log_pfa.plot(
            log10_pfa, pd, 
            label=f'{label_name}',
            color=colors[i], 
            linestyle=linestyles[i], 
            linewidth=1
        )
        if rx:
            ax_pfa_thresh_rx.plot(
                thresholds, pfa, 
                label=f'{label_name}',
                color=colors[i], 
                linestyle=linestyles[i], 
                linewidth=1
            )
            ax_log_pfa_thresh_rx.plot(
                thresholds, log10_pfa, 
                label=f'{label_name}', 
                color=colors[i], 
                linestyle=linestyles[i], 
                linewidth=1
            )
        else:
            ax_pfa_thresh_cvae.plot(
                thresholds, pfa, 
                label=f'{label_name}',
                color=colors[i], 
                linestyle=linestyles[i], 
                linewidth=1
            )
            ax_log_pfa_thresh_cvae.plot(
                thresholds, log10_pfa, 
                label=f'{label_name}', 
                color=colors[i], 
                linestyle=linestyles[i], 
                linewidth=1
            )
    # Finalize and save plots
    ax_roc.set_xlabel('Probability of False Alarm')
    ax_roc.set_ylabel('Probability of Detection')
    ax_roc.legend()
    ax_roc.grid()
    ax_roc.figure.savefig(f'{opt.save_dir}/pd-pfa.png', dpi=300)
    
    ax_roc_log_pfa.set_xlabel('Log Probability of False Alarm')
    ax_roc_log_pfa.set_ylabel('Probability of Detection')
    ax_roc_log_pfa.set_xlim([-4.5, 0])
    ax_roc_log_pfa.legend()
    ax_roc_log_pfa.grid()
    ax_roc_log_pfa.figure.savefig(f'{opt.save_dir}/pd-logpfa.png', dpi=300)
    
    ax_pfa_thresh_rx.set_xlabel('Thresholds')
    ax_pfa_thresh_rx.set_ylabel('Probability of False Alarm')
    ax_pfa_thresh_rx.legend()
    ax_pfa_thresh_rx.grid()
    ax_pfa_thresh_rx.figure.savefig(f'{opt.save_dir}/pfa-thresholds_rx.png', dpi=300)
    
    ax_log_pfa_thresh_rx.set_xlabel('Thresholds')
    ax_log_pfa_thresh_rx.set_ylabel('Log Probability of False Alarm')
    ax_log_pfa_thresh_rx.legend()
    ax_log_pfa_thresh_rx.grid()
    ax_log_pfa_thresh_rx.figure.savefig(f'{opt.save_dir}/logpfa-thresholds_rx.png', dpi=300)

    ax_pfa_thresh_cvae.set_xlabel('Thresholds')
    ax_pfa_thresh_cvae.set_ylabel('Probability of False Alarm')
    ax_pfa_thresh_cvae.legend()
    ax_pfa_thresh_cvae.grid()
    ax_pfa_thresh_cvae.figure.savefig(f'{opt.save_dir}/pfa-thresholds_cvae.png', dpi=300)
    
    ax_log_pfa_thresh_cvae.set_xlabel('Thresholds')
    ax_log_pfa_thresh_cvae.set_ylabel('Log Probability of False Alarm')
    ax_log_pfa_thresh_cvae.legend()
    ax_log_pfa_thresh_cvae.grid()
    ax_log_pfa_thresh_cvae.figure.savefig(f'{opt.save_dir}/logpfa-thresholds_cvae.png', dpi=300)
    
    plt.close('all')


def plot_graphs_synthetic_anomalies(opt: Namespace, ano_type: str, snr: float) -> None:
    # Create subplots
    _, ax_roc = plt.subplots()
    _, ax_roc_log_pfa = plt.subplots()
    _, ax_pfa_thresh_rx = plt.subplots()
    _, ax_log_pfa_thresh_rx = plt.subplots()
    _, ax_pfa_thresh_cvae = plt.subplots()
    _, ax_log_pfa_thresh_cvae = plt.subplots()
    # Define colors and linestyles
    colors = ['blue', 'red', 'green', 'cyan', 'orange']
    linestyles = ['-', '--', ':', '-.', '-']
    # Check if enough colors are defined
    if len(opt.anomaly_map_dirs) > len(colors):
        raise ValueError('Not enough colors defined for the number of anomaly map directories provided.')
    # First filtering: only RX-SCM, RX-Tyler, Frobenius; synthetic; correct snr and ano_type; no real_valued
    paths_anomaly_map = [
        path for anomaly_map_dir in opt.anomaly_map_dirs 
        for path in glob.glob(f'{anomaly_map_dir}/*.npy')
        if ('RX-SCM' in path or 'frobenius' in path or 'RX-Tyler' in path) and ('synthetic' in path) and (f'snr_{snr}' in path) and (f'anomalies_{ano_type}' in path) and ('real_valued' not in path)
    ]
    # Second filtering: remove boxcar for RX-SCM and all shift variants
    paths_anomaly_map = [
        path for path in paths_anomaly_map if (not (('RX-SCM' in path) and ('boxcar' in path))) and ('shift' not in path)
    ]
    # Check if the number of found anomaly map files matches the expected count. +2 for RX-Tyler with and without boxcar
    if len(paths_anomaly_map) != len(opt.anomaly_map_dirs) + 2:
        raise ValueError('Number of found anomaly map files does not match the number of provided directories.')
    # Initialize lists to store PD at fixed PFA and SNR values
    pd_at_fixed_pfa = {}
    # Iterate over each anomaly map path and compute ROC curves
    for i, pam in enumerate(paths_anomaly_map):
        # Determine if cropping is needed
        crop = any(sub in Path(pam).stem for sub in ['RX', 'frobenius'])
        # Determine if RX method
        rx = ('RX-SCM' in Path(pam).stem) or ('RX-Tyler' in Path(pam).stem)
        # Determine boxcar size for each method
        if 'frobenius' in Path(pam).stem:
            bcs = int(os.path.dirname(pam)[-1])
        elif 'boxcar' in Path(pam).stem:
            bcs = int(Path(pam).stem.split('boxcar')[-1].split('_')[1][0])
        else:
            bcs = 31
        # Get clutter path
        path_clutter = pam.replace('synthetic_noisy_anomalies', 'clutter_noisy_crop').replace(f'{ano_type}_snr_{snr}_', '')
        # Load data
        clutter, anomaly_map, label = get_anomaly_map_and_label_synthetic_anomalies(opt, pam, path_clutter, bcs, crop, rx)
        # Compute ROC
        pd, pfa, thresholds = compute_roc_synthetic_anomalies(anomaly_map, clutter, label)
        # Determine label name. ComplexVAE or Reed-Xiaoli
        label_name = 'complexVAE' if any(sub in Path(pam).stem for sub in ['frobenius', 'manhattan', 'euclidean']) else 'RX'
        # RX-Tyler or RX-SCM
        label_name = label_name + '-Tyler' if 'Tyler' in Path(pam).stem else label_name + '-SCM' if 'RX-SCM' in Path(pam).stem else label_name
        # Append boxcar size if applicable
        label_name = label_name + f' - Boxcar {bcs}' if bcs != 31 else label_name
        # Store PD at fixed PFA
        idx_fixed_pfa = int(len(pfa) * opt.pfa_for_pd_snr)
        pd_at_fixed_pfa[label_name] = pd[idx_fixed_pfa]
        # Convert pfa to log10 scale
        log10_pfa = np.log10(pfa + 1e-10)
        # Plotting
        ax_roc.plot(
            pfa, pd, 
            label=f'{label_name} - AUC: {auc(pfa, pd):.3f}',
            color=colors[i], 
            linestyle=linestyles[i], 
            linewidth=1
        )
        ax_roc_log_pfa.plot(
            log10_pfa, pd, 
            label=f'{label_name}',
            color=colors[i], 
            linestyle=linestyles[i], 
            linewidth=1
        )
        if rx:
            ax_pfa_thresh_rx.plot(
                thresholds, pfa, 
                label=f'{label_name}',
                color=colors[i], 
                linestyle=linestyles[i], 
                linewidth=1
            )
            ax_log_pfa_thresh_rx.plot(
                thresholds, log10_pfa, 
                label=f'{label_name}', 
                color=colors[i], 
                linestyle=linestyles[i], 
                linewidth=1
            )
        else:
            ax_pfa_thresh_cvae.plot(
                thresholds, pfa, 
                label=f'{label_name}',
                color=colors[i], 
                linestyle=linestyles[i], 
                linewidth=1
            )
            ax_log_pfa_thresh_cvae.plot(
                thresholds, log10_pfa, 
                label=f'{label_name}', 
                color=colors[i], 
                linestyle=linestyles[i], 
                linewidth=1
            )
    # Finalize and save plots
    ax_roc.set_xlabel('Probability of False Alarm')
    ax_roc.set_ylabel('Probability of Detection')
    ax_roc.legend()
    ax_roc.grid()
    ax_roc.figure.savefig(f'{opt.save_dir}/pd-pfa_snr_{snr:.1f}_{ano_type}.png', dpi=300)
    
    ax_roc_log_pfa.set_xlabel('Log Probability of False Alarm')
    ax_roc_log_pfa.set_ylabel('Probability of Detection')
    ax_roc_log_pfa.set_xlim([-4.5, 0])
    ax_roc_log_pfa.legend()
    ax_roc_log_pfa.grid()
    ax_roc_log_pfa.figure.savefig(f'{opt.save_dir}/pd-logpfa_snr_{snr:.1f}_{ano_type}.png', dpi=300)
    
    ax_pfa_thresh_rx.set_xlabel('Thresholds')
    ax_pfa_thresh_rx.set_ylabel('Probability of False Alarm')
    ax_pfa_thresh_rx.legend()
    ax_pfa_thresh_rx.grid()
    ax_pfa_thresh_rx.figure.savefig(f'{opt.save_dir}/pfa-thresholds_rx_snr_{snr:.1f}_{ano_type}.png', dpi=300)
    
    ax_log_pfa_thresh_rx.set_xlabel('Thresholds')
    ax_log_pfa_thresh_rx.set_ylabel('Log Probability of False Alarm')
    ax_log_pfa_thresh_rx.legend()
    ax_log_pfa_thresh_rx.grid()
    ax_log_pfa_thresh_rx.figure.savefig(f'{opt.save_dir}/logpfa-thresholds_rx_snr_{snr:.1f}_{ano_type}.png', dpi=300)

    ax_pfa_thresh_cvae.set_xlabel('Thresholds')
    ax_pfa_thresh_cvae.set_ylabel('Probability of False Alarm')
    ax_pfa_thresh_cvae.legend()
    ax_pfa_thresh_cvae.grid()
    ax_pfa_thresh_cvae.figure.savefig(f'{opt.save_dir}/pfa-thresholds_cvae_snr_{snr:.1f}_{ano_type}.png', dpi=300)
    
    ax_log_pfa_thresh_cvae.set_xlabel('Thresholds')
    ax_log_pfa_thresh_cvae.set_ylabel('Log Probability of False Alarm')
    ax_log_pfa_thresh_cvae.legend()
    ax_log_pfa_thresh_cvae.grid()
    ax_log_pfa_thresh_cvae.figure.savefig(f'{opt.save_dir}/logpfa-thresholds_cvae_snr_{snr:.1f}_{ano_type}.png', dpi=300)
    
    plt.close('all')

    return pd_at_fixed_pfa


def plot_pd_snr(opt: Namespace, pd_at_fixed_pfa: Sequence[dict], ano_type: str) -> None:
    out = defaultdict(list)
    for d in pd_at_fixed_pfa:
        for k, v in d.items():
            out[k].append(v)
    
    out = dict(out)
    # Plotting PD vs SNR at fixed PFA
    for k, v in out.items():
        plt.plot(
            np.linspace(1, 20, dtype=np.float32, num=20), 
            v, 
            label=k,
            marker='o',
            linestyle='-',
            linewidth=1.5,
            markersize=3,
        )
    plt.xlabel('SNR (dB)')
    plt.ylabel(f'Probability of Detection')
    plt.legend()
    plt.grid()
    plt.savefig(f'{opt.save_dir}/pd_snr_at_pfa_{opt.pfa_for_pd_snr}_{ano_type}.png', dpi=300)
    plt.close('all')


def compute_curves(opt: Namespace, ano_type: str) -> None:
    pd_at_fixed_pfa_all = []
    for a_snr in tqdm(np.linspace(1, 20, dtype=np.float32, num=20), ascii=' >', desc=f'Computing ROC curves for {ano_type}: '):
        pd_at_fixed_pfa_all.append(plot_graphs_synthetic_anomalies(opt, ano_type, a_snr))

    plot_pd_snr(opt, pd_at_fixed_pfa_all, ano_type)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path_label', type=str, required=True, help='Path of the ground truth label file.')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to save plots.')
    parser.add_argument('--anomaly_map_dirs', nargs='+', required=True, help='Directories of the anomaly map files.')
    parser.add_argument('--pfa_for_pd_snr', type=float, default=0.02, help='Fixed PFA value to compute PD vs SNR.')
    parser.add_argument('--dso_boxcar', type=int, default=9, help='Boxcar size for DSO dataset if applicable.')
    opt = parser.parse_args()

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    
    if 'DSO' not in opt.path_label:
        for at in ['white', 'violet', 'green']:
            compute_curves(opt, at)
    else:
        plot_graphs_labeled_anomalies(opt)