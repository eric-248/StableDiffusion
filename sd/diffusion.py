import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    
    def __init__(self, n_embd: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.linear_1(x)

        x = F.silu(x)

        x = self.linear_2(x)

        return x
    
class Upsample(nn.Module):
    
    def __init__(self, channels:int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # (BS, Features, H, W) -> (BS, F, H * 2, W * 2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

class SwitchSequential(nn.Sequential):

    def forward(self, x:torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context) #cross attention between latents and prompts
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time) #match with time step
            else:
                x = layer(x)

        return x


    
class UNET(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoders = nn.Module([

            #(BS, 4, H/8, W/8)    
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            #(BS, 320, H/8, W/8) -> (BS, 320, H/16, W/16) 
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            #(BS, 640, H/16, W/16) -> (BS, 640, H/32, W/32) 
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            #(BS, 640, H/32, W/32) -> (BS, 640, H/64, W/64) 
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280)),

            #(BS, 1280, H/64, W/64)  
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        
        ])  

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),

            UNET_AttentionBlock(8, 160),

            UNET_ResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList([

            #(BS, 2560, H/64, W/64) -> (BS, 1280, H/64, W/64)
            #doubles because of skip connections
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            # (BS, 1280, H/64, W/64) -> (BS, 1280, H/32, W/32), Upsample(num_heads, dim_per_head)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),

            #add cross attention using 8 heads of 160 dim
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            #encoder skip is 640 channels: 1280 + 640 = 1920
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)), 

            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            
            #Decoder (640) + skip(320) = 960
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),

            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),


        ])

class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels:int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (BS, 320, H/8, W/8)

        x = self.groupnorm(x)

        x = F.silu(x)

        x = self.conv(x)

        # (BS, 4, H/8, W/8)
        return x

class Diffusion(nn.Module):

    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time):
        #latent: (BS, 4, H/8, W/8)
        #context: (BS, Seq_Len, Dim)
        #time: (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (BS, 4, H/8, W/8) -> (BS, 320, H/8, W/8)
        output = self.unet(latent, context, time)

        # (BS, 320, H/8, W/8) -> (BS, 4, H/8, W/8)
        output = self.final(output)

        return output

