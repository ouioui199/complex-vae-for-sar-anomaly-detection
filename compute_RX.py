from typing import Sequence
from pathlib import Path
from argparse import ArgumentParser, Namespace
import glob, sys, os

from PIL import Image
import numpy as np
from numba import njit, prange
from tqdm import tqdm

sys.path.append('./scripts')
sys.path.append('./scripts/datasets')

from scripts.utils import ArgumentParsing
from scripts.datasets.utils import ensure_chw_format


@njit(fastmath=True)
def diag_nb(A: np.ndarray) -> np.ndarray:
    """Extract the diagonal elements of a matrix.
    
    Numba-compatible implementation of np.diag() for extracting diagonal elements.
    
    Args:
        A: Input matrix of shape (m, n)
    
    Returns:
        1D array containing diagonal elements [A[0,0], A[1,1], ..., A[k,k]] 
        where k = min(m, n)
    """
    n = min(A.shape[0], A.shape[1])
    out = np.empty(n, dtype=A.dtype)
    for i in range(n):
        out[i] = A[i, i]
    return out


@njit(fastmath=True)
def mean_axis1_keepdims(a: np.ndarray) -> np.ndarray:
    """Compute mean along axis 1 while keeping dimensions.
    
    Numba-compatible implementation of np.mean(a, axis=1, keepdims=True).
    Computes the mean of each row independently.
    
    Args:
        a: Input 2D array of shape (n, m)
    
    Returns:
        2D array of shape (n, 1) containing the mean of each row
    """
    n, m = a.shape
    out = np.zeros((n, 1), dtype=a.dtype)
    for i in range(n):
        s = 0.
        for j in range(m):
            s += a[i, j]
        out[i, 0] = s / m
    return out


@njit(fastmath=True)
def compute_complex_valued_mean_cov(rx_type: str, test_pixel: np.ndarray, background: np.ndarray) -> float:
    # Compute covariance with regularization
    if rx_type == 'scm':
        no_shift_cov = (background @ np.conj(background.T)) / background.shape[1]
    elif rx_type == 'tyler':
        no_shift_cov, _, _ = tyler_estimator_covariance(background)
    # Invert covariance matrix
    inv_no_shift_cov = np.linalg.inv(no_shift_cov)
    # Compute RX detector
    no_shift_rx_value = (np.conj(test_pixel.T) @ inv_no_shift_cov @ test_pixel)[0, 0].real
    return no_shift_rx_value


@njit(fastmath=True)
def compute_real_valued_mean_cov(test_pixel: np.ndarray, background: np.ndarray) -> Sequence[float]:
    # Compute mean and centered background
    mu = mean_axis1_keepdims(background)
    centered_bg = background - mu
    # Compute covariance with regularization
    shift_cov = (centered_bg @ centered_bg.T) / (background.shape[1] - 1)
    no_shift_cov = (background @ background.T) / (background.shape[1] - 1)
    # Invert covariance matrices
    inv_no_shift_cov = np.linalg.inv(no_shift_cov)
    inv_shift_cov = np.linalg.inv(shift_cov)
    # Center test pixel
    centered_test = test_pixel - mu
    # Compute RX detector
    shift_rx_value = (centered_test.T @ inv_shift_cov @ centered_test)[0, 0]
    no_shift_rx_value = (test_pixel.T @ inv_no_shift_cov @ test_pixel)[0, 0]
    return no_shift_rx_value, shift_rx_value


@njit(fastmath=True)
def get_background(
    image: np.ndarray,
    pixel_coordinates: Sequence[int],
    exclusion_window_size: int,
    box_car_size: int,
    guard_window: bool = True
) -> Sequence[np.ndarray]:
    c = image.shape[0]
    # Extract test pixel
    x, y = pixel_coordinates
    test_pixel = image[:, x, y].copy().reshape(c, 1)
    # Define background region boundaries
    half_box = box_car_size // 2
    half_excl = exclusion_window_size // 2
    # Define box car region
    x_start = x - half_box
    x_end = x + half_box + 1
    y_start = y - half_box
    y_end = y + half_box + 1
    # Extract background pixels
    if guard_window:
        # Extract box car region
        box_car = image[:, x_start:x_end, y_start:y_end]
        # Create mask with 1s (include) and 0s (exclude)
        mask = np.ones((box_car_size, box_car_size), dtype=np.bool_)
        # Set exclusion window to 0
        excl_start_x = half_box - half_excl
        excl_end_x = half_box + half_excl + 1
        excl_start_y = half_box - half_excl
        excl_end_y = half_box + half_excl + 1
        mask[excl_start_x:excl_end_x, excl_start_y:excl_end_y] = 0
        # Reshape box_car to (c, box_car_size*box_car_size)
        box_car_flat = box_car.copy().reshape(c, -1)
        # Flatten mask and use it to filter background pixels
        mask_flat = mask.flatten()
        background = box_car_flat[:, mask_flat]
    else:
        # Extract background region directly
        background = image[:, x_start:x_end, y_start:y_end].copy().reshape(c, -1)

    return test_pixel, background

@njit(fastmath=True)
def tyler_estimator_covariance(𝐗: np.ndarray, tol: float = 0.0001, iter_max: int = 100) -> Sequence[np.ndarray | int]:
    """ A function that computes the Tyler Fixed Point Estimator for covariance matrix estimation
        Inputs:
            * 𝐗 = a matrix of size p*N with each observation along column dimension
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * 𝚺 = the estimate
            * δ = the final distance between two iterations
            * iteration = number of iterations til convergence """
    
    # Initialisation
    (p, N) = 𝐗.shape
    δ = np.inf # Distance between two iterations
    𝚺 = np.eye(p, dtype=𝐗.dtype) # Initialise estimate to identity
    iteration = 0
    # Recursive algorithm
    while (δ > tol) and (iteration < iter_max):
        # Computing expression of Tyler estimator (with matrix multiplication)
        τ = diag_nb(𝐗.conj().T @ np.linalg.inv(𝚺) @ 𝐗)
        # Ensure proper dtype preservation
        sqrt_tau = np.sqrt(τ)
        𝐗_bis = np.ascontiguousarray((𝐗 / sqrt_tau).astype(𝐗.dtype))
        # Compute Hermitian conjugate as contiguous array
        𝐗_bis_H = np.ascontiguousarray(𝐗_bis.conj().T.astype(𝐗.dtype))
        # Matrix multiplication with matching dtypes
        𝚺_new = (p/N) * (𝐗_bis @ 𝐗_bis_H)
        # Imposing trace constraint: Tr(𝚺) = p
        𝚺_new = p * 𝚺_new / np.trace(𝚺_new)
        # Condition for stopping
        δ = np.linalg.norm(𝚺_new - 𝚺) / np.linalg.norm(𝚺)
        iteration = iteration + 1
        # Updating 𝚺
        𝚺 = 𝚺_new.astype(𝐗.dtype)

    if iteration == iter_max:
        print('Recursive algorithm did not converge')
    
    return (𝚺, δ, iteration)


@njit(parallel=True, fastmath=True)
def compute_cplx_reed_xiaoli_detector(
    rx_type: str,
    h: int, 
    w: int, 
    image: np.ndarray, 
    rx_exclusion_window_size: int, 
    rx_box_car_size: int,
    rx_guard_window: bool
) -> np.ndarray:
    """Generic function to compute Reed-Xiaoli detector maps using the specified detector function."""
    no_shift_output= np.zeros((h, w), dtype=np.float32)
    # Compute RX values for each pixel (excluding image borders)
    half_box = rx_box_car_size // 2
    for i in prange(half_box, h - half_box):
        for j in range(half_box, w - half_box):
            test_pixel, background = get_background(
                image, (i, j), rx_exclusion_window_size, rx_box_car_size, rx_guard_window
            )
            no_shift_output[i, j] = compute_complex_valued_mean_cov(rx_type, test_pixel, background)
    
    return no_shift_output


@njit(parallel=True, fastmath=True)
def compute_real_reed_xiaoli_detector(
    h: int, 
    w: int, 
    image: np.ndarray, 
    rx_exclusion_window_size: int, 
    rx_box_car_size: int,
    rx_guard_window: bool
) -> Sequence[np.ndarray]:
    """Generic function to compute Reed-Xiaoli detector maps using the specified detector function."""
    no_shift_output, shift_output = np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)
    # Compute RX values for each pixel (excluding image borders)
    half_box = rx_box_car_size // 2
    for i in prange(half_box, h - half_box):
        for j in range(half_box, w - half_box):
            test_pixel, background = get_background(
                image, (i, j), rx_exclusion_window_size, rx_box_car_size, rx_guard_window
            )
            no_shift_output[i, j], shift_output[i, j] = compute_real_valued_mean_cov(test_pixel, background)
    
    return no_shift_output, shift_output
        

def get_reed_xiaoli_map(opt: Namespace) -> None:
    datadir = Path(f'{opt.datadir}/{opt.data_band}_band/predict/slc')
    paths = glob.glob(f'{datadir}/*.npy')
    # path_rx = [p for p in paths if 'clutter' in p]
    path_rx = [p for p in paths if ('Combine' in p) and ('synthetic' in os.path.basename(p)) and ('mask' not in os.path.basename(p)) and ('crop' not in os.path.basename(p))]
    # path_rx = [p for p in paths if (('euclidean' in p) or ('manhattan' in p)) and ('sum' not in os.path.basename(p))]
    if not path_rx:
        print('No combined polarization image found. Computing combined polarization image...')
        paths = glob.glob(f'{datadir}/*.npy')
        path_rx = [p for p in paths if 'Combine' in p]
    # Determine if guard window is to be used
    rx_guard_window = True if opt.rx_exclusion_window_size != 0 else False
    # Retrieve RX parameters
    rx_type = opt.rx_type
    rx_exclusion_window_size = opt.rx_exclusion_window_size
    rx_box_car_size = opt.rx_box_car_size
    rx_real_valued = opt.rx_real_valued
    # Define affixes for saving RX maps
    affix0 = 'RX-SCM' if rx_type == 'scm' else 'RX-Tyler'
    affix1 = '_real_valued' if rx_real_valued else ''
    affix2 = f'_no_guard_window_boxcar_{rx_box_car_size}' if not rx_guard_window else ''
    # Compute RX maps
    for p_rx in tqdm(path_rx, ascii=' >', desc='Computing Reed-Xiaoli RX maps'):
        # Load image
        image = np.load(p_rx, mmap_mode='r')
        # Ensure the image is in CHW format
        image = ensure_chw_format(image)
        c, h, w = image.shape
        if rx_real_valued:
            image = np.abs(image)
            no_shift_output, shift_output = compute_real_reed_xiaoli_detector(
                h, w, image, rx_exclusion_window_size, rx_box_car_size, rx_guard_window
            )
            # Crop borders
            shift_output = shift_output[
                np.newaxis, 
                rx_box_car_size // 2 : -rx_box_car_size // 2, 
                rx_box_car_size // 2 : -rx_box_car_size // 2
            ]
            # Save RX maps
            np.save(p_rx.replace('Combined', affix0 + '-shift' + affix1 + affix2), shift_output)
            # Save as images for visualization
            shift_output = Image.fromarray(((shift_output.squeeze() / np.max(shift_output)) * 255).astype(np.uint8))
            shift_output.save(p_rx.replace('Combined', affix0 + '-shift' + affix1 + affix2).replace('.npy', '.png'))
        else:
            no_shift_output = compute_cplx_reed_xiaoli_detector(
                rx_type, h, w, image, rx_exclusion_window_size, rx_box_car_size, rx_guard_window
            )
        # Crop borders
        no_shift_output = no_shift_output[
            np.newaxis, 
            rx_box_car_size // 2 : -rx_box_car_size // 2, 
            rx_box_car_size // 2 : -rx_box_car_size // 2
        ]
        # Save RX maps
        np.save(p_rx.replace('Combined', affix0 + affix1 + affix2), no_shift_output)
        # Save as images for visualization
        no_shift_output = Image.fromarray(((no_shift_output.squeeze() / np.max(no_shift_output)) * 255).astype(np.uint8))
        no_shift_output.save(p_rx.replace('Combined', affix0 + affix1 + affix2).replace('.npy', '.png'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = ArgumentParsing(parser)
    opt = parser.parser.parse_args()
    #NOTE: for each file having 4 polarization channels, we need to manually combine them into a single file having 'Combine' in its name.
    # To do so, run the combine.ipynb notebook. Change the directory if needed.
    # These RX files must be computed only one time, I don't want to put energy into thinking how to do this properly...
    get_reed_xiaoli_map(opt)
