import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            #(BS, CH, H, W) -> (BS, 128, H, W)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            #(BS, 128, H, W) -> (BS, 128, H, W)
            VAE_ResidualBlock(128, 128),

            #(BS, 128, H, W) -> (BS, 128, H, W)
            VAE_ResidualBlock(128, 128),

            #(BS, 128, H, W) -> (BS, 128, H/2, W/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding = 0),

            #(BS, 128, H, W) -> (BS, 256, H, W)
            VAE_ResidualBlock(128, 256),

            #(BS, 256, H, W) -> (BS, 256, H, W)
            VAE_ResidualBlock(256, 256),

            #(BS, 256, H/2, W/2) -> (BS, 256, H/4, W/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding = 0),

            #(BS, 256, H/4, W/4) -> (BS, 256, H/4, W/4)
            VAE_ResidualBlock(256, 512),

            #(BS, 512, H/4, W/4) -> (BS, 512, H/4, W/4)
            VAE_ResidualBlock(512, 512),

            #(BS, 512, H/4, W/4) -> (BS, 512, H/8, W/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding = 0),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            #(BS, 512, H/8, W/8) -> (BS, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),

            #(BS, 512, H/8, W/8) -> (BS, 512, H/8, W/8)
            VAE_AttentionBlock(512),

            #(BS, 512, H/8, W/8) -> (BS, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),

            #(BS, 512, H/8, W/8) -> (BS, 512, H/8, W/8)
            nn.GroupNorm(32, 512),

            #(BS, 512, H/8, W/8) -> (BS, 512, H/8, W/8)
            nn.SiLU(),

            #(BS, 512, H/8, W/8) -> (BS, 8, H/8, W/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            #(BS, 8, H/8, W/8) -> (BS, 8, H/8, W/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)

        )

    def forward(self, x: torch.Tensor, noise:torch.Tensor) -> torch.Tensor:
        #x : (BS, CH, H, W)
        #noise: (BS, Output_CH, H/8, W/8)

        #if downsampling layer, apply padding
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                #(Padding_Left, Padding_Right, Padding_Top, Padding_Bottom)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # (BS, 8, H/8, W/8) -> two tensors of shape (BS, 4, H/8, W/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # (BS, 4, H/8, W/8) -> (BS, 4, H/8, W/8)
        log_variance = torch.clamp(log_variance, -30, 20)

        # (BS, 4, H/8, W/8) -> (BS, 4, H/8, W/8)
        variance = log_variance.exp()

         # (BS, 4, H/8, W/8) -> (BS, 4, H/8, W/8)
        stdev = variance.sqrt()

        # Z=N(0,1) -> N(mean, variance)?
        # X = mean + stdev * Z
        x = mean + stdev * noise

        #Scale the output by a constant
        x *= 0.18215

        return x
