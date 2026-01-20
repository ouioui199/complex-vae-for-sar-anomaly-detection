import os

import numpy as np


def ensure_chw_format(x: np.ndarray) -> np.ndarray:
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
