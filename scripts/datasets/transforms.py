from typing import Sequence
from types import ModuleType

import numpy as np
import torch
from torch import Tensor


class MinMaxNormalize:
    
    def __init__(self, min: np.ndarray, max: np.ndarray) -> None:
        self.min = min
        self.max = max

    def call_real(self, image: np.ndarray) -> np.ndarray:
        log_image = np.log(np.abs(image) + np.spacing(1))
        normalized_image = (log_image - self.min) / (self.max - self.min)
        normalized_image = np.clip(normalized_image, -1, 1)
        return normalized_image

    def call_complex(self, image: np.ndarray) -> np.ndarray:
        magnitude, phase = np.abs(image), np.angle(image)
        
        log_magnitude = np.log(magnitude + np.spacing(1))
        normalized_magnitude = (log_magnitude - self.min) / (self.max - self.min)
        normalized_magnitude = np.clip(normalized_magnitude, 0, 1)
        
        return normalized_magnitude * np.exp(1j * phase)

    def __call__(self, image: np.ndarray) -> Sequence[np.ndarray]:
        if np.iscomplexobj(image):
            return self.call_complex(image)
        else:
            return self.call_real(image)
    

class MinMaxDenormalize:

    def __init__(self, min: np.ndarray | Tensor, max: np.ndarray | Tensor) -> None:
        self.min = min
        self.max = max

    def _cast(self, x: np.ndarray | Tensor, backend: ModuleType) -> np.ndarray | Tensor:
        if backend is np:
            return x.astype(np.float32, copy=False)
        elif backend is torch:
            return x.to(torch.float32)
        else:
            raise ValueError("Backend must be either numpy or torch")

    def call_real(self, norm_image: np.ndarray | Tensor, backend: ModuleType) -> np.ndarray | Tensor:
        log_image = (self.max - self.min) * norm_image + self.min
        return backend.exp(self._cast(log_image, backend)) - np.spacing(1)

    def call_complex(self, norm_image: np.ndarray | Tensor, backend: ModuleType) -> np.ndarray | Tensor:
        norm_image, phase = (backend.abs(norm_image), backend.angle(norm_image))
        log_image = (self.max - self.min) * norm_image + self.min

        return (backend.exp(self._cast(log_image, backend)) - np.spacing(1)) * backend.exp(1j * phase)

    def denorm_ndarray(self, norm_image: np.ndarray) -> np.ndarray:
        if np.iscomplexobj(norm_image):
            return self.call_complex(norm_image, np)
        else:
            return self.call_real(norm_image, np)

    def denorm_tensor(self, norm_image: Tensor) -> Tensor:
        if torch.is_complex(norm_image):
            return self.call_complex(norm_image, torch)
        else:
            return self.call_real(norm_image, torch)
    
    def __call__(self, norm_image: np.ndarray | Tensor) -> np.ndarray | Tensor:
        if isinstance(norm_image, np.ndarray):
            return self.denorm_ndarray(norm_image)
        elif isinstance(norm_image, Tensor):
            return self.denorm_tensor(norm_image)
