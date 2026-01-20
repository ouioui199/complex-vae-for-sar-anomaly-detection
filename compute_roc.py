from pathlib import Path
from typing import Sequence
from argparse import ArgumentParser, Namespace
from itertools import product
import glob, os

import numpy as np
import numba as nb
from sklearn.metrics import auc
from matplotlib import pyplot as plt
from tqdm import tqdm


def get_anomaly_map_and_label(
    opt: Namespace, 
    path_anomaly_map: str, 
    path_clutter: str, 
    box_car_size: int, 
    crop: bool, rx: bool
) -> Sequence[np.ndarray]:
    # Load label and convert to tensor once
    label = np.load(opt.path_label, mmap_mode='r')
    # Load and preprocess anomaly map
    anomaly_map = np.load(str(Path(path_anomaly_map)))
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
    # Load clutter for thresholding
    clutter = np.load(path_clutter, mmap_mode='r').flatten()
    return clutter, anomaly_map, label


def compute_roc(anomaly_map: np.ndarray, clutter: np.ndarray, label: np.ndarray) -> Sequence[np.ndarray]:
    clutter = np.sort(clutter)

    @nb.jit(parallel=True, nopython=True)
    def compute_all_pd_pfa(anomaly_map, clutter, label):
        pd = np.zeros(len(clutter))
        pfa = np.zeros(len(clutter))
        thresholds = np.zeros(len(clutter))
        
        for i in nb.prange(1, len(clutter) + 1):
            threshold_value = clutter[- i]
            ano_map_pfa_threshold = np.where(anomaly_map >= threshold_value, 1, 0)
                
            tp = ((ano_map_pfa_threshold == 1) & (label == 1)).sum()
            fn = ((ano_map_pfa_threshold == 0) & (label == 1)).sum()
            
            pd[i - 1] = tp / (tp + fn)
            pfa[i - 1] = i / len(clutter)
            thresholds[i - 1] = threshold_value
        
        return pd, pfa, thresholds
    
    pd, pfa, thresholds = compute_all_pd_pfa(anomaly_map, clutter, label)
    return pd, pfa, thresholds


def plot_graphs(opt: Namespace, ano_type: str, snr: float) -> None:
    _, ax_roc = plt.subplots()
    _, ax_roc_log_pfa = plt.subplots()
    _, ax_pfa_thresh_rx = plt.subplots()
    _, ax_log_pfa_thresh_rx = plt.subplots()
    _, ax_pfa_thresh_cvae = plt.subplots()
    _, ax_log_pfa_thresh_cvae = plt.subplots()
    
    colors = ['blue', 'red', 'green']
    linestyles = ['-', '--', ':']

    if len(opt.anomaly_map_dirs) > len(colors):
        raise ValueError('Not enough colors defined for the number of anomaly map directories provided.')

    paths_anomaly_map = [
        path for anomaly_map_dir in opt.anomaly_map_dirs 
        for path in glob.glob(f'{anomaly_map_dir}/*.npy')
        if ('RX-SCM-shift' in path or 'frobenius' in path) and ('synthetic' in path) and (f'snr_{snr}' in path) and (f'anomalies_{ano_type}' in path) and ('boxcar' not in path) and ('real_valued' not in path)
    ]
    
    if len(paths_anomaly_map) != len(opt.anomaly_map_dirs):
        raise ValueError('Number of found anomaly map files does not match the number of provided directories.')

    for i, pam in enumerate(paths_anomaly_map):
        crop = any(sub in Path(pam).stem for sub in ['RX-SCM', 'frobenius'])
        rx = 'RX-SCM' in Path(pam).stem
        bcs = os.path.dirname(pam)[-1]
        bcs = int(bcs) if bcs.isdigit() else 31
        path_clutter = pam.replace('synthetic_noisy_anomalies', 'clutter_noisy_crop').replace(f'{ano_type}_snr_{snr}_', '')
        clutter, anomaly_map, label = get_anomaly_map_and_label(opt, pam, path_clutter, bcs, crop, rx)
        pd, pfa, thresholds = compute_roc(anomaly_map, clutter, label)
        label_name = 'complexVAE' if any(sub in Path(pam).stem for sub in ['frobenius', 'manhattan', 'euclidean']) else 'Reed-Xiaoli SCM'
        label_name = label_name + ' Tyler' if 'real_valued' in Path(pam).stem else label_name
        
        log10_pfa = np.log10(pfa + 1e-10)
        ax_roc.plot(
            pfa, pd, 
            label=f'{label_name} - Boxcar {bcs} - AUC: {auc(pfa, pd):.3f}' if not rx else f'{label_name} - AUC: {auc(pfa, pd):.3f}',
            color=colors[i], 
            linestyle=linestyles[i], 
            linewidth=1
        )
        ax_roc_log_pfa.plot(
            log10_pfa, pd, 
            label=f'{label_name} - Boxcar {bcs} - {snr} dB' if not rx else f'{label_name}',
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
                label=f'{label_name} - Boxcar {bcs} - {snr} dB', 
                color=colors[i], 
                linestyle=linestyles[i], 
                linewidth=1
            )
            ax_log_pfa_thresh_cvae.plot(
                thresholds, log10_pfa, 
                label=f'{label_name} - Boxcar {bcs} - {snr} dB', 
                color=colors[i], 
                linestyle=linestyles[i], 
                linewidth=1
            )
        
    ax_roc.set_xlabel('Probability of False Alarm')
    ax_roc.set_ylabel('Probability of Detection')
    ax_roc.set_title('PD - PFA Curves Comparison')
    ax_roc.legend()
    ax_roc.grid()
    ax_roc.figure.savefig(f'{opt.save_dir}/pd-pfa_snr_{snr:.1f}_{ano_type}.png', dpi=300)
    
    ax_roc_log_pfa.set_xlabel('Log Probability of False Alarm')
    ax_roc_log_pfa.set_ylabel('Probability of Detection')
    ax_roc_log_pfa.set_title('PD - Log(PFA) Curves Comparison')
    ax_roc_log_pfa.set_xlim([-4.5, 0])
    ax_roc_log_pfa.legend()
    ax_roc_log_pfa.grid()
    ax_roc_log_pfa.figure.savefig(f'{opt.save_dir}/pd-logpfa_snr_{snr:.1f}_{ano_type}.png', dpi=300)
    
    ax_pfa_thresh_rx.set_xlabel('Thresholds')
    ax_pfa_thresh_rx.set_ylabel('Probability of False Alarm')
    ax_pfa_thresh_rx.set_title('PFA - Thresholds Comparison')
    ax_pfa_thresh_rx.legend()
    ax_pfa_thresh_rx.grid()
    ax_pfa_thresh_rx.figure.savefig(f'{opt.save_dir}/pfa-thresholds_rx_snr_{snr:.1f}_{ano_type}.png', dpi=300)
    
    ax_log_pfa_thresh_rx.set_xlabel('Thresholds')
    ax_log_pfa_thresh_rx.set_ylabel('Log Probability of False Alarm')
    ax_log_pfa_thresh_rx.set_title('Log(PFA) - Thresholds Comparison')
    ax_log_pfa_thresh_rx.legend()
    ax_log_pfa_thresh_rx.grid()
    ax_log_pfa_thresh_rx.figure.savefig(f'{opt.save_dir}/logpfa-thresholds_rx_snr_{snr:.1f}_{ano_type}.png', dpi=300)

    ax_pfa_thresh_cvae.set_xlabel('Thresholds')
    ax_pfa_thresh_cvae.set_ylabel('Probability of False Alarm')
    ax_pfa_thresh_cvae.set_title('PFA - Thresholds Comparison')
    ax_pfa_thresh_cvae.legend()
    ax_pfa_thresh_cvae.grid()
    ax_pfa_thresh_cvae.figure.savefig(f'{opt.save_dir}/pfa-thresholds_cvae_snr_{snr:.1f}_{ano_type}.png', dpi=300)
    
    ax_log_pfa_thresh_cvae.set_xlabel('Thresholds')
    ax_log_pfa_thresh_cvae.set_ylabel('Log Probability of False Alarm')
    ax_log_pfa_thresh_cvae.set_title('Log(PFA) - Thresholds Comparison')
    ax_log_pfa_thresh_cvae.legend()
    ax_log_pfa_thresh_cvae.grid()
    ax_log_pfa_thresh_cvae.figure.savefig(f'{opt.save_dir}/logpfa-thresholds_cvae_snr_{snr:.1f}_{ano_type}.png', dpi=300)
    
    plt.close('all')
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path_label', type=str, required=True, help='Path of the ground truth label file.')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to save plots.')
    parser.add_argument('--anomaly_map_dirs', nargs='+', required=True, help='Directories of the anomaly map files.')
    opt = parser.parse_args()

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    
    for at, a_snr in tqdm(
        product(
            ['white', 'violet', 'green'], 
            np.linspace(1, 20, dtype=np.float32, num=20),
        ), ascii=' >', desc='Computing ROC curves: ', total=60):
        plot_graphs(opt, at, a_snr)
