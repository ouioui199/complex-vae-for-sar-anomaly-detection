from typing import Sequence

import torch
from torch import nn, Tensor
from torch.nn.functional import softplus
from torchcvnn import nn as c_nn
from torchcvnn.nn.modules import initialization as init


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        activation: nn.Module, 
        in_kernel: int = 3, 
        in_padding: int = 1, 
        stride: int = 1, 
        mid_channels: int = None
    ) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=in_kernel,
                stride=stride,
                padding=in_padding,
                # bias=False,
                padding_mode="replicate",
                dtype=torch.complex64,
            ),
            c_nn.BatchNorm2d(
                mid_channels,
                track_running_stats=False
            ),
            activation,
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                # bias=False,
                padding_mode="replicate",
                dtype=torch.complex64,
            ),
            c_nn.BatchNorm2d(
                out_channels,
                track_running_stats=False
            ),
            activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        activation: nn.Module
    ) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConv(
                in_channels,
                out_channels,
                activation,
                stride=2,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        activation: nn.Module
    ) -> None:
        super().__init__()
        self.up = c_nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(out_channels, out_channels, activation)

    def forward(self, x: Tensor) -> Tensor:
        x = self.up(x)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, dtype=torch.complex64),
        )
        # self.conv_reconstruction = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=1, dtype=torch.complex64),
        # )
        # self.conv_classification = nn.Sequential(
        #     nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, dtype=torch.complex64),
        # )

    def forward(self, x: Tensor) -> Tensor:
        reconstruction = self.conv(x)
        # reconstruction = self.conv_reconstruction(x)
        if not self.training:
            return reconstruction
            
        # classification = torch.abs(self.conv_classification(x))
        return reconstruction, None #, classification


class AutoEncoder(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_layers: int,
        channels_ratio: int,
        activation: nn.Module,
    ):
        super().__init__()
        self.n_channels = num_channels
        # Encoder with doubling channels
        current_channels = channels_ratio
        self.encoder_layers = []
        self.encoder_layers.append(
            DoubleConv(self.n_channels, current_channels, activation, in_kernel=9, in_padding=4)
        )
        for i in range(1, num_layers):
            out_channels = channels_ratio * 2**i
            self.encoder_layers.append(Down(current_channels, out_channels, activation))
            current_channels = out_channels
        # Encoder
        self.encoder_out_channels = current_channels
        self.encoder_layers = nn.Sequential(*self.encoder_layers)
        # Decoder
        self.decoder_layers = []
        current_channels = self.encoder_out_channels
        for i in range(num_layers - 2, -1, -1):
            out_channels = channels_ratio * 2**i
            self.decoder_layers.append(Up(current_channels, out_channels, activation))
            current_channels = out_channels
        self.decoder_layers.append(OutConv(current_channels, num_channels))
        self.decoder_layers = nn.Sequential(*self.decoder_layers)

    @staticmethod
    def initialize_network(module: nn.Module) -> None:
        for m in module.children():
            if isinstance(m, nn.Conv2d):
                init.complex_kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.Sequential):
                # Recursively initialize submodules
                AutoEncoder.initialize_network(m)

    def initialize_weights(self) -> None:
        self.initialize_network(self.encoder_layers)
        self.initialize_network(self.decoder_layers)

    def forward(self, input: Tensor) -> Tensor:
        z = self.encoder_layers(input)
        return self.decoder_layers(z)
        
        
class complexVAE(AutoEncoder):
    def __init__(
        self,
        num_channels: int,
        num_layers: int,
        channels_ratio: int,
        activation: nn.Module,
        latent_compression: int
    ):
        super().__init__(num_channels, num_layers, channels_ratio, activation)
        self.mu = nn.Conv2d(
            self.encoder_out_channels,
            self.encoder_out_channels // latent_compression,
            kernel_size=1,
            stride=1,
            dtype=torch.complex64
        )
        self.sigma = nn.Conv2d(
            self.encoder_out_channels,
            self.encoder_out_channels // latent_compression,
            kernel_size=1,
            stride=1,
            dtype=torch.complex64
        )
        self.delta = nn.Conv2d(
            self.encoder_out_channels,
            self.encoder_out_channels // latent_compression,
            kernel_size=1,
            stride=1,
            dtype=torch.complex64
        )
        if latent_compression > 1:
            self.decoder_input = nn.Conv2d(
                self.encoder_out_channels // latent_compression,
                self.encoder_out_channels,
                kernel_size=1,
                stride=1,
                dtype=torch.complex64
            )

    def initialize_weights(self) -> None:
        super().initialize_weights()
        self.initialize_network(self.mu)
        self.initialize_network(self.sigma)
        self.initialize_network(self.delta)
        if hasattr(self, 'decoder_input'):
            self.initialize_network(self.decoder_input)
        
    def encode(self, input: Tensor) -> Sequence[Tensor]:
        result = self.encoder_layers(input)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.mu(result)
        sigma = self.sigma(result)
        delta = self.delta(result)
        return mu, sigma, delta
    
    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        if hasattr(self, 'decoder_input'):
            z = self.decoder_input(z)
        return self.decoder_layers(z)
    
    def reparameterize_delta(self, sigma: Tensor, raw_delta: Tensor, eps: float = 1e-7) -> Tensor:
        """
        Contrainte: delta = alpha*sigma, avec |alpha| < 1
        => |delta| < sigma
        sigma     : [batch_size, H] (réel, > 0)
        raw_delta : [batch_size, H] (complex)
        """
        # On prend la norme du raw_delta
        mag = torch.sqrt(raw_delta.real ** 2 + raw_delta.imag ** 2 + eps)
        # On définit un "facteur" = mag/(1+mag) < 1
        factor = mag / (1.0 + mag)
        # Calcul des composantes de alpha
        alpha_r = torch.where(mag > eps, raw_delta.real * (factor / mag), torch.zeros_like(mag))
        alpha_i = torch.where(mag > eps, raw_delta.imag * (factor / mag), torch.zeros_like(mag))
        # => alpha = alpha_r + i alpha_i, de module < 1
        alpha = torch.complex(alpha_r, alpha_i)
        # On multiplie par sigma (réel > 0)
        delta = alpha * sigma

        return delta

    def reparameterize(self, mu: Tensor, sigma: Tensor, delta: Tensor) -> Tensor:
        delta = self.reparameterize_delta(sigma, delta)

        k_x = (sigma + delta) / torch.sqrt(2 * (sigma + delta.real))
        k_y = torch.sqrt((sigma ** 2 - torch.abs(delta) ** 2) / (2 * (sigma + delta.real)))
        
        eps_x = torch.randn_like(k_x, dtype=torch.float32)
        eps_y = torch.randn_like(k_y, dtype=torch.float32)
        
        return mu + k_x * eps_x + 1j * k_y * eps_y, delta

    def forward(self, input: Tensor) -> Sequence[Tensor]:
        mu, sigma, delta = self.encode(input)
        sigma = softplus(sigma.real) + 1e-6
        z, delta = self.reparameterize(mu, sigma, delta)
        return z, mu, sigma, delta
