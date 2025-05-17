import torch 
from torch import nn 

class UNetGenerator(nn.Module):
    """U-Net generator with skip connections (for Pix2Pix/CycleGAN)."""
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d):
        super().__init__()
        
        # Initial convolution (no downsampling yet)
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        # Downsampling (reduce spatial size, increase channels)
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        # Residual blocks (bottleneck)
        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [ResidualBlock(ngf * mult, norm_layer=norm_layer)]

        # Upsampling (increase spatial size, reduce channels)
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                      norm_layer(ngf * mult // 2),
                      nn.ReLU(True)]

        # Final output (tanh for [-1, 1] range)
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):
    """Residual block with skip connection (used in U-Net bottleneck)."""
    def __init__(self, dim, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            norm_layer(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            norm_layer(dim))
        
    def forward(self, x):
        return x + self.block(x)  # Skip connection