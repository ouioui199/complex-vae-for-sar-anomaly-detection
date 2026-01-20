from pathlib import Path
from types import NoneType
from typing import Sequence
from itertools import product
import os

import numpy as np
from tqdm import tqdm
from PIL import Image
from torchcvnn.transforms.functional import equalize


class SyntheticAnomalyGenerator:
    """Class for generating synthetic anomalies in SAR images."""

    def __init__(
        self, 
        noisy_path: str,
        output_dir: str,
        location: Sequence[int],
        base_ano_center: Sequence[int],
        ano_size: Sequence[int],
        ano_snr: float,
        ano_type: str
    ) -> None:
        """Initialize with file paths."""
        self.noisy_path = str(Path(noisy_path))
        self.output_dir = str(Path(output_dir))
        self.location = location
        self.base_ano_center = base_ano_center
        self.ano_size = ano_size
        self.configs = None
        self.mask = None
        self.ano_snr = ano_snr
        self.ano_type = ano_type

        if self.ano_type not in ['violet', 'green', 'white']:
            raise ValueError(f"Anomaly type {self.ano_type} not recognized. Choose from 'violet', 'green', 'white'.")
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        if not os.path.exists(self.noisy_path):
            raise FileNotFoundError(f"Noisy file {self.noisy_path} does not exist.")
        
        # Define steering vectors
        self.steering_vector = {
            'violet': np.array([1, 0, 0, 1]),
            'green': np.array([0, 1, 1, 0]),
            'white': np.array([1, 1, 1, 1]),
        }
    
    def load_data(self) -> None:
        """Load and preprocess noisy data."""
        self.noisy = np.load(self.noisy_path)
        self.noisy = self.noisy[:, self.location[0] : self.location[0] + 256, self.location[1] : self.location[1] + 256].transpose(1,2,0)

    def get_anomaly_configs(self) -> None:
        """Define anomaly configurations for different colors and positions."""
        # Compute covariance matrix
        noisy_reshaped = self.noisy.reshape(-1, self.noisy.shape[-1]).T  # shape (C, H*W)
        covariance_noisy = noisy_reshaped @ np.conj(noisy_reshaped).T / noisy_reshaped.shape[1] # shape (C, C)
        # Compute anomaly values based on SNR
        p_vector = self.steering_vector[self.ano_type]
        noisy_anomaly_value = np.sqrt(
            10 ** (self.ano_snr / 10) / (np.conj(p_vector).T @ np.linalg.inv(covariance_noisy) @ p_vector).real
        )
        # Generate configurations for anomalies
        self.configs = []
        for i, j in product(range(1, 5), repeat=2):
            self.configs.append({
                'value': noisy_anomaly_value * p_vector,
                'ano_center': (self.base_ano_center[0] * i, self.base_ano_center[1] * j),
                'ano_size': (self.ano_size[0], self.ano_size[1])
            })
    
    def generate_anomalies(self) -> None:
        """Generate synthetic anomalies using the configurations."""
        if self.configs is None:
            self.get_anomaly_configs()

        for config in self.configs:
            self.mask = self.add_rect(mask=self.mask, **config)

    def add_rect(self, value: np.ndarray, ano_center: Sequence[int], ano_size: Sequence[int], mask: np.ndarray | NoneType, replace: bool = False) -> np.ndarray:
        """Add rectangular anomaly to noisy images."""
        shape = self.noisy.shape

        if mask is None:
            mask = np.zeros((shape[0], shape[1]))
        assert mask.shape == (shape[0], shape[1])

        ano_half_height, ano_half_width = ano_size[0] // 2, ano_size[1] // 2
        ano_center_height, ano_center_width = ano_center[0], ano_center[1]
        # Define rectangle regions (horizontal and vertical bars)
        rect_regions = [
            # Horizontal rectangle
            (slice(ano_center_height - ano_half_height, ano_center_height + ano_half_height),
            slice(ano_center_width - ano_half_width, ano_center_width + ano_half_width)),
            # Vertical rectangle
            (slice(ano_center_height - ano_half_width, ano_center_height + ano_half_width),
            slice(ano_center_width - ano_half_height, ano_center_width + ano_half_height))
        ]
        # Apply anomaly to both mask and reconstruction
        for row_slice, col_slice in rect_regions:
            mask[row_slice, col_slice] = 1
            if replace:
                self.noisy[row_slice, col_slice, :] = value
            else:
                self.noisy[row_slice, col_slice, :] += value

        return mask
    
    def save_visualizations(self) -> None:
        """Save visualizations of the anomalous image and mask."""
        noisy_path = os.path.join(self.output_dir, f'synthetic_noisy_anomalies_{self.ano_type}_snr_{self.ano_snr}_' + os.path.basename(self.noisy_path))
        np.save(noisy_path, self.noisy.transpose(2, 0, 1))
        self.noisy = equalize(self.noisy, pupper=100, plower=0)
        self.noisy = np.stack((self.noisy[..., 0], self.noisy[..., 1], self.noisy[..., 3]), axis=2)
        self.noisy = Image.fromarray(self.noisy)
        self.noisy.save(noisy_path.replace('.npy', '.png'), dpi=(300, 300))

        mask_path = os.path.join(self.output_dir, 'anomaly_mask_' + os.path.basename(self.noisy_path))
        np.save(mask_path, self.mask)
        self.mask = Image.fromarray((self.mask * 255).astype(np.uint8))
        self.mask.save(mask_path.replace('.npy', '.png'), dpi=(300, 300))

    def launch(self) -> None:
        """Launch the full anomaly generation process."""
        # Load data
        self.load_data()
        # Generate anomalies
        self.generate_anomalies()
        # Save visualizations
        self.save_visualizations()
    

if __name__ == '__main__':
    
    # File paths
    noisy_path = ... # specify your noisy input file path here
    output_dir = ... # specify your output directory here
    # Generate brights anomalies with various SNRs
    for at, ap in tqdm(
        product(
            ['white', 'violet', 'green'], 
            np.linspace(1, 20, dtype=np.float32, num=20)
        ), ascii=' >', desc='Generating anomaly configurations', total=60):
        generator = SyntheticAnomalyGenerator(
            noisy_path,
            output_dir,
            location=(73, 0),
            base_ano_center=(51, 51),
            ano_size=(25, 4),
            ano_snr=ap,
            ano_type=at
        )
        generator.launch()
