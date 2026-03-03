from typing import Sequence, Tuple
from types import ModuleType
from argparse import Namespace
from pathlib import Path
import os

import numpy as np
import scipy
import jax.numpy as jnp
from jax import Array
from scipy import signal


def ensure_chw_format(x: np.ndarray | Array) -> np.ndarray | Array:
    """Ensure image is in CHW format, convert if necessary.
    
    Args:
        x (np.ndarray): Input image to check/convert format
        
    Returns:
        np.ndarray: Image in CHW format
        
    Raises:
        TypeError: If input is not numpy array or torch tensor
        ValueError: If input is not a 3D array
        
    Example:
        >>> img = np.zeros((64, 64, 3))  # HWC format
        >>> chw_img = ensure_chw_format(img)  # Converts to (3, 64, 64)
    """
    if len(x.shape) != 3:
        raise ValueError("Image must be 3D array")
    # Convert from HWC to CHW, channel is often the smallest dimension in a SAR image.
    if min(x.shape) != x.shape[0]:
        return x.transpose(2, 0, 1)
    return x


def is_directory_empty(directory_path: str) -> bool:
    """Check if a directory is empty.
    
    Args:
        directory_path: Path to the directory to check
        
    Returns:
        True if the directory is empty or doesn't exist, False otherwise
    """
    if not os.path.exists(directory_path):
        return True
        
    if os.path.isdir(directory_path):
        # Check if directory contains any files or subdirectories
        return len(os.listdir(directory_path)) == 0
    
    # If it's not a directory, return False
    return False


def process_dot_mat(opt: Namespace, filepath: str, backend: ModuleType) -> np.ndarray | Array:
    if backend not in [np, jnp]:
        raise ValueError("Unsupported backend. Please use numpy or jax.numpy.")
    # Load DSO .mat file
    mat_data = scipy.io.loadmat(filepath)
    # Get crop coordinates based on non-black regions across all polarizations
    crop_coords = [get_non_black_coordinates(mat_data[pol], backend=backend) for pol in ['hh', 'hv', 'vh', 'vv']]
    crop_coords = backend.stack(crop_coords, axis=1)
    # Get final crop coordinates that encompass all non-black regions
    r0, r1, c0, c1, c1_diff = get_crop_coordinates(crop_coords, backend=backend)
    data = backend.stack([
        mat_data['hh'][r0:r1, c0:c1],
        mat_data['hv'][r0:r1, c0:c1],
        mat_data['vh'][r0:r1, c0 + c1_diff:c1 + c1_diff],
        mat_data['vv'][r0:r1, c0:c1]
    ], axis=0)
    # Process label
    ano_pixels = mat_data['pixels'][0]
    ano_pixels = [p for p in ano_pixels if isinstance(p, backend.ndarray) and p.size > 0]
    if len(ano_pixels) == 0:
        ano_pixels = backend.empty((0, 2), dtype=backend.int32)
    else:
        ano_pixels = backend.concatenate(ano_pixels, axis=0, dtype=backend.int32)
    # Adjust anomaly pixel coordinates based on cropping
    if backend is jnp:
        ano_pixels = ano_pixels.at[:, 0].add(-r0)
        ano_pixels = ano_pixels.at[:, 1].add(-c0)
    else:
        ano_pixels[:, 0] -= r0
        ano_pixels[:, 1] -= c0
    # Safe detection to add
    safe_pixels = 5
    # Process oversampling if needed
    if opt.undersample_dso:
        _, h, w = data.shape
        data = process_oversampled_image(
            data,
            ground_range_res=1.232,
            ground_azimuth_res=0.6,
            row_pixel_size=0.36,
            col_pixel_size=0.36,
            backend=backend
        )
        filepath = filepath.replace('.mat', '_undersampled.npy')
        # Adjust anomaly pixel coordinates based on undersampling
        _, h_new, w_new = data.shape
        ano_pixels = (ano_pixels * np.array([[h_new / h, w_new / w]])).astype(backend.int32)
        safe_pixels = 2
    # Create label with safe detection zone
    label_filepath = filepath.replace('.npy', '_label.npy')
    if not Path(label_filepath).exists():
        label = backend.zeros_like(data[0], dtype=backend.uint8)[backend.newaxis, :, :]
        for row, col in ano_pixels:
            upper_row = max(0, row - safe_pixels)
            lower_row = min(label.shape[1], row + safe_pixels + 1)
            upper_col = max(0, col - safe_pixels)
            lower_col = min(label.shape[2], col + safe_pixels + 1)
            if backend is jnp:
                label = label.at[:, upper_row:lower_row, upper_col:lower_col].set(1)
            else:
                label[:, upper_row:lower_row, upper_col:lower_col] = 1
        # Save label
        backend.save(label_filepath, label[0])
    else:
        label = backend.load(label_filepath)

    return data, filepath, label


def get_non_black_coordinates(image: np.ndarray | Array, backend: ModuleType) -> np.ndarray | Array:
    image = backend.abs(image)
    # non-black mask
    mask = image > 0
    # find bounding box
    rows = backend.where(mask.any(axis=1))[0]
    cols = backend.where(mask.any(axis=0))[0]
    # crop the image
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    
    return backend.array([r0, r1, c0, c1], dtype=backend.int32)


def get_crop_coordinates(coords: np.ndarray | Array, backend: ModuleType) -> np.ndarray | Array:
    r0 = backend.max(coords[0])
    r1 = backend.min(coords[1])
    c0 = backend.max(coords[2])
    c1 = backend.min(coords[3])
    # There is a constant shift in azimuth direction on the VH polarization on DSO dataset. 
    # We compute the difference here for later adjustment.
    c1_diff = backend.max(coords[3]) - backend.min(coords[3])
    
    return r0, r1, c0, c1, c1_diff


def process_oversampled_image(
        image: np.ndarray | Array, 
        ground_range_res: float, 
        ground_azimuth_res: float,
        row_pixel_size: float,
        col_pixel_size: float,
        backend: ModuleType
    ) -> np.ndarray | Array:
    _, h, w = image.shape
    fft = backend.fft.fftshift(backend.fft.fft2(image, axes=(-2, -1)), axes=(-2, -1))
    fft_crop = fft[
        :,
        int(h // 2 - (h / (ground_range_res / row_pixel_size)) // 2): int(h // 2 + (h / (ground_range_res / row_pixel_size)) // 2),
        int(w // 2 - (w / (ground_azimuth_res / col_pixel_size)) // 2): int(w // 2 + (w / (ground_azimuth_res / col_pixel_size)) // 2)
    ]
    
    return backend.fft.ifft2(backend.fft.ifftshift(fft_crop, axes=(-2, -1)), axes=(-2, -1))


def get_data_in_pauli_decomposition(data: np.ndarray | Array, backend: ModuleType) -> np.ndarray | Array:
    if data.shape[0] != 4:
        raise ValueError("Input data must have 4 channels corresponding to HH, HV, VH, VV polarizations.")
    
    hh, hv, vh, vv = data
    alpha = (hh + vv) / np.sqrt(2)
    beta = (hh - vv) / np.sqrt(2)
    gamma = (hv + vh) / np.sqrt(2)
    return backend.stack([beta, gamma, alpha], axis=0)


def combine_hv_vh(data: np.ndarray | Array, backend: ModuleType) -> np.ndarray | Array:
    if data.shape[0] != 4:
        raise ValueError("Input data must have 4 channels corresponding to HH, HV, VH, VV polarizations.")
    
    hh, hv, vh, vv = data
    return backend.stack([hh, (hv + vh) / 2, vv], axis=0)


def process_image_representation(image: np.ndarray | Array, backend: ModuleType, opt: Namespace) -> np.ndarray | Array:
    if opt.channels_type == 'slc' and opt.in_channels == 3:
        image = combine_hv_vh(image, backend)
    elif opt.channels_type == 'pauli':
        image = get_data_in_pauli_decomposition(image, backend)

    return np.abs(image) if 'cplx' not in opt.recon_model else image