from typing import Sequence
from pathlib import Path
from argparse import ArgumentParser, Namespace
from functools import partial
import glob, sys

import jax.numpy as jnp
from jax import lax, Array, jit, vmap, device_get
from PIL import Image
from tqdm import tqdm

sys.path.append('./scripts')
sys.path.append('./scripts/datasets')

from scripts.utils import ArgumentParsing, combine_polar_channels, is_valid_rx_file
from scripts.datasets.utils import (
    ensure_chw_format,
    process_dso_mat,
    process_image_representation
)


@partial(jit, static_argnames=("rx_type"))
def compute_complex_valued_mean_cov(rx_type: str, test_pixel: Array, background: Array) -> float:
    # Compute covariance with regularization
    if rx_type == 'scm':
        no_shift_cov = (background @ jnp.conj(background.T)) / background.shape[1]
        converged = True
    elif rx_type == 'tyler':
        (no_shift_cov, _, _), converged = tyler_estimator_covariance(background)
    # Invert covariance matrix
    inv_no_shift_cov = jnp.linalg.inv(no_shift_cov)
    # Compute RX detector
    no_shift_rx_value = (jnp.conj(test_pixel.T) @ inv_no_shift_cov @ test_pixel)[0, 0].real
    return no_shift_rx_value, converged


@jit
def compute_real_valued_mean_cov(test_pixel: Array, background: Array) -> Sequence[float]:
    # Compute mean and centered background
    mu = jnp.mean(background, axis=1, keepdims=True)
    centered_bg = background - mu
    # Compute covariance with regularization
    shift_cov = (centered_bg @ centered_bg.T) / (background.shape[1] - 1)
    no_shift_cov = (background @ background.T) / (background.shape[1] - 1)
    # Invert covariance matrices
    inv_no_shift_cov = jnp.linalg.inv(no_shift_cov)
    inv_shift_cov = jnp.linalg.inv(shift_cov)
    # Center test pixel
    centered_test = test_pixel - mu
    # Compute RX detector
    shift_rx_value = (centered_test.T @ inv_shift_cov @ centered_test)[0, 0]
    no_shift_rx_value = (test_pixel.T @ inv_no_shift_cov @ test_pixel)[0, 0]
    return no_shift_rx_value, shift_rx_value


@partial(jit, static_argnames=('rx_exclusion_window_size', 'rx_box_car_size', 'guard_window'))
def get_background(
    image: Array,
    pixel_coordinates: Sequence[int],
    rx_exclusion_window_size: int,
    rx_box_car_size: int,
    guard_window: bool = True
) -> Sequence[Array]:
    c = image.shape[0]
    # Extract test pixel
    x, y = pixel_coordinates
    test_pixel = image[:, x, y].reshape(c, 1)
    # Define background region boundaries
    half_box = rx_box_car_size // 2
    half_excl = rx_exclusion_window_size // 2
    # Extract box car region
    box_car = lax.dynamic_slice(
        image,
        (0, x - half_box, y - half_box),
        (c, rx_box_car_size, rx_box_car_size),
    )
    # Extract background pixels
    if guard_window:
        # Create mask with 1s (include) and 0s (exclude)
        mask = jnp.ones((rx_box_car_size, rx_box_car_size), dtype=bool)
        # Set exclusion window to 0
        excl_start = half_box - half_excl
        excl_end = half_box + half_excl + 1
        mask = mask.at[
            excl_start:excl_end,
            excl_start:excl_end
        ].set(False)
        # Reshape box_car to (c, box_car_size*box_car_size)
        box_car_flat = box_car.reshape(c, -1)
        # Flatten mask and use it to filter background pixels
        mask_flat = mask.reshape(-1)
        num_bg = rx_box_car_size * rx_box_car_size - rx_exclusion_window_size * rx_exclusion_window_size
        idx = jnp.nonzero(mask_flat, size=mask_flat.size, fill_value=0)[0][:num_bg]
        background = jnp.take(box_car_flat, idx, axis=1)
    else:
        # Extract background region directly
        background = box_car.reshape(c, -1)

    return test_pixel, background


@jit
def tyler_estimator_covariance(𝐗: Array, tol: float = 3e-4, iter_max: int = 100) -> Sequence[Array | int]:
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
    𝚺 = jnp.eye(p, dtype=𝐗.dtype) # Initialise estimate to identity
    # Conditional function
    def cond_fun(val: Sequence[Array | int]) -> bool:
        _, δ, iteration = val
        return (δ > tol) & (iteration < iter_max)
    # Body function
    def body_fun(val: Sequence[Array | int]) -> Sequence[Array | int]:
        Σ, δ, iteration = val
        # Computing expression of Tyler estimator (with matrix multiplication)
        τ = jnp.diag(𝐗.conj().T @ jnp.linalg.inv(𝚺) @ 𝐗)
        𝐗_bis = 𝐗 / jnp.sqrt(τ)
        𝚺_new = (p/N) * (𝐗_bis @ 𝐗_bis.conj().T)
        # Imposing trace constraint: Tr(𝚺) = p
        𝚺_new = p * 𝚺_new / jnp.trace(𝚺_new)
        # Condition for stopping
        δ = jnp.linalg.norm(𝚺_new - 𝚺) / jnp.linalg.norm(𝚺)
        iteration = iteration + 1
        
        return 𝚺_new, δ, iteration
    
    𝚺_final, δ_final, iteration_final = lax.while_loop(cond_fun, body_fun, (𝚺, jnp.inf, 0))
    
    return (𝚺_final, δ_final, iteration_final), iteration_final < iter_max


def compute_cplx_reed_xiaoli_detector(
    rx_type: str,
    h: int, 
    w: int, 
    image: Array,
    rx_exclusion_window_size: int,
    rx_box_car_size: int,
    rx_guard_window: bool,
    batch_size: int
) -> Array:
    half_box = rx_box_car_size // 2
    # Generate pixel coordinates
    i_coords = jnp.arange(half_box, h - half_box)
    j_coords = jnp.arange(half_box, w - half_box)
    I, J = jnp.meshgrid(i_coords, j_coords, indexing='ij')
    coords = jnp.stack([I.flatten(), J.flatten()], axis=1)
    # Output array
    no_shift_output = jnp.zeros((h, w), dtype=jnp.float32)
    # Helper function for a chunk of coordinates
    def compute_pixel(coord: Array) -> float:
        i, j = coord
        test_pixel, background = get_background(
            image, (i, j), rx_exclusion_window_size, rx_box_car_size, rx_guard_window
        )
        return compute_complex_valued_mean_cov(rx_type, test_pixel, background)
    # Vectorized computation using vmap
    non_converged_pixels = 0
    for start in range(0, len(coords), batch_size):
        end = start + batch_size
        results = vmap(compute_pixel)(coords[start:end])
        no_shift_output = no_shift_output.at[coords[start:end, 0], coords[start:end, 1]].set(results[0])
        non_converged_pixels += len(results[1]) - jnp.sum(results[1])

    print('\nNumber of Tyler non-converged pixels:', non_converged_pixels)
    return no_shift_output


def compute_real_reed_xiaoli_detector(
    h: int, 
    w: int, 
    image: Array, 
    rx_exclusion_window_size: int, 
    rx_box_car_size: int,
    rx_guard_window: bool,
    batch_size: int
) -> Sequence[Array]:
    """Generic function to compute Reed-Xiaoli detector maps using the specified detector function."""
    half_box = rx_box_car_size // 2
    # Generate pixel coordinates
    i_coords = jnp.arange(half_box, h - half_box)
    j_coords = jnp.arange(half_box, w - half_box)
    I, J = jnp.meshgrid(i_coords, j_coords, indexing='ij')
    coords = jnp.stack([I.flatten(), J.flatten()], axis=1)
    # Output array
    no_shift_output, shift_output = jnp.zeros((h, w), dtype=jnp.float32), jnp.zeros((h, w), dtype=jnp.float32)
    # Helper function for a chunk of coordinates
    def compute_pixel(coord: Array) -> float:
        i, j = coord
        test_pixel, background = get_background(
            image, (i, j), rx_exclusion_window_size, rx_box_car_size, rx_guard_window
        )
        return compute_real_valued_mean_cov(test_pixel, background)
    # Vectorized computation using vmap
    for start in range(0, len(coords), batch_size):
        end = start + batch_size
        results = vmap(compute_pixel)(coords[start:end])
        no_shift_output = no_shift_output.at[coords[start:end, 0], coords[start:end, 1]].set(results[0])
        shift_output = shift_output.at[coords[start:end, 0], coords[start:end, 1]].set(results[1])
    
    return no_shift_output, shift_output


def get_reed_xiaoli_map(opt: Namespace) -> None:
    datadir = Path(f'{opt.datadir}/{opt.data_band}_band/predict/slc')
    paths = glob.glob(f'{datadir}/*.npy') if 'DSO' not in str(datadir) else glob.glob(f'{datadir}/*.mat')
    path_rx = [p for p in paths if is_valid_rx_file(p, opt.rx_data)]
    if not path_rx:
        print('No combined polarization image found. Computing combined polarization image...')
        combine_polar_channels(paths)
        paths = glob.glob(f'{datadir}/*.npy')
        path_rx = [p for p in paths if is_valid_rx_file(p, opt.rx_data)]
    # Determine if guard window is to be used
    rx_guard_window = True if opt.rx_exclusion_window_size != 0 else False
    # Retrieve RX parameters
    rx_type = opt.rx_type
    rx_exclusion_window_size = opt.rx_exclusion_window_size
    rx_box_car_size = opt.rx_box_car_size
    rx_real_valued = opt.rx_real_valued
    batch_size = opt.rx_chunk_batch_size
    # Define affixes for saving RX maps
    affix0 = 'RX-SCM' if rx_type == 'scm' else 'RX-Tyler'
    affix0 +=  + '-shift' if rx_real_valued else ''
    affix1 = '_real_valued' if rx_real_valued else ''
    affix2 = f'_no_guard_window_boxcar_{rx_box_car_size}' if not rx_guard_window else f'_{rx_exclusion_window_size}_{rx_box_car_size}'
    # Compute RX maps
    for p_rx in tqdm(path_rx, ascii=' >', desc='Computing Reed-Xiaoli RX maps'):
        # Load image
        if 'DSO' in p_rx:
            image, p_rx, _ = process_dso_mat(opt, p_rx, jnp)
            p_rx = p_rx.replace('.mat', '.npy')
            p_rx = str(Path(p_rx).with_name(Path(p_rx).stem + '_' + affix0 + affix1 + affix2 + Path(p_rx).suffix))
        else:
            image = jnp.load(p_rx, mmap_mode='r')
        # Ensure the image is in CHW format
        image = ensure_chw_format(image)
        c, h, w = image.shape
        # Process complex image to more sophisticated representation if needed
        image = process_image_representation(image, jnp, opt)
        p_rx = p_rx.replace('Combined', affix0 + affix1 + affix2)
        if rx_real_valued:
            image = jnp.abs(image)
            no_shift_output, shift_output = compute_real_reed_xiaoli_detector(
                h, w, image, rx_exclusion_window_size, rx_box_car_size, rx_guard_window, batch_size
            )
            # Crop borders
            shift_output = shift_output[
                jnp.newaxis, 
                rx_box_car_size // 2 : -rx_box_car_size // 2, 
                rx_box_car_size // 2 : -rx_box_car_size // 2
            ]
            # Save RX maps
            jnp.save(p_rx, shift_output)
            # Save as images for visualization
            shift_output = device_get(((shift_output.squeeze() / jnp.max(shift_output)) * 255).astype(jnp.uint8))
            shift_output = Image.fromarray(shift_output)
            shift_output.save(p_rx.replace('.npy', '.png'))
        else:
            no_shift_output = compute_cplx_reed_xiaoli_detector(
                rx_type, h, w, image, rx_exclusion_window_size, rx_box_car_size, rx_guard_window, batch_size
            )
        # Crop borders
        no_shift_output = no_shift_output[
            jnp.newaxis, 
            rx_box_car_size // 2 : -rx_box_car_size // 2, 
            rx_box_car_size // 2 : -rx_box_car_size // 2
        ]
        # Save RX maps
        jnp.save(p_rx, no_shift_output)
        # Save as images for visualization
        no_shift_output = device_get(((no_shift_output.squeeze() / jnp.max(no_shift_output)) * 255).astype(jnp.uint8))
        no_shift_output = Image.fromarray(no_shift_output)
        no_shift_output.save(p_rx.replace('.npy', '.png'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = ArgumentParsing(parser)
    opt = parser.parser.parse_args()
    
    get_reed_xiaoli_map(opt)
