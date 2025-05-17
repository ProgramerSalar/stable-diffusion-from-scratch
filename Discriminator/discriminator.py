
import torch 
from torch import nn 
import functools


class ActNorm(nn.Module):

    def __init__(self,
                 num_features,
                 logdet=False,
                 affine=True,
                 allow_reverse_init=False):
        

        assert affine, "make sure affine method is `True`"
        super().__init__()
        self.logdet = logdet

        # Learnable parameters 
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1)) # shift parameters
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1)) # Scale parameters
        self.allow_reverse_init = allow_reverse_init

        # track whether initialize has occupied 
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))


    def initialize(self, input):

        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)


            # compute mean and std per channels 
            mean = (
                flatten.mean(1)
                .unsqueeze(1).unsqueeze(2).unsqueeze(2) # [1, C, 1, 1]
                .permute(1, 0, 2, 3)
            )

            std = (
                flatten.mean(1)
                .unsqueeze(1).unsqueeze(2).unsqueeze(2) # [1, C, 1, 1]
                .permute(1, 0, 2, 3)
            )

            # Initialize parameters 
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))


    def forward(self, input, reverse=False):

        if reverse:
            return self.reverse(output=input)
        

        # Handle 2D input (e.g [B, C], [B, C, 1, 1])
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True

        else:
            squeeze = False 

        _, _, height, width = input.shape 

        # initialize on first foward pass in training 
        if self.training and self.initialized.item == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        # Apply normalization 
        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)


        # Compute log determinant for flows 
        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet
        

        return h 
    


        

    def reverse(self, output):

        """Reverse pass: denormalize the output."""

        if self.training and self.initialize.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initialized ActNorm is reverse direction is disabled by default. `allow_reverse_init=True` to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)


        # handle 2d inputs 
        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True

        else:
            squeeze = False 


        # Denomalize: x = (h / scale) - loc 
        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        return h 


        


        





class Discriminator(nn.Module):

    def __init__(self,
                 input_nc=3, 
                 ndf=64,
                 n_layers=3,
                 use_actnorm=False):
        
        super().__init__()
        

        # choose normalization layer
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d   
        else:
            norm_layer = ActNorm 

        # Determine if conv layer should use bias terms 
        if isinstance(norm_layer, functools.partial):
            use_bias = (norm_layer.func != nn.BatchNorm2d) # NO bias for BatchNorm

        else:
            use_bias = (norm_layer != nn.BatchNorm2d)

        


        kw = 4 # kernel size 
        padw = 1 # padding 

        # downsampling layer 
        sequence = [nn.Conv2d(in_channels=input_nc,
                              out_channels=ndf, 
                              kernel_size=kw,
                              stride=2,
                              padding=padw),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        

        # Gradually increase filter count (up to 8x ndf)
        nf_mult = 1 # number of filter mulltiplier 
        nf_mult_prev = 1  # number of filter multipier previous 
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)

            sequence += [
                nn.Conv2d(in_channels=ndf * nf_mult_prev,
                          out_channels=ndf * nf_mult,
                          kernel_size=kw,
                          stride=2,
                          padding=padw,
                          bias=use_bias),
                norm_layer(num_features=ndf * nf_mult),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ]


            
        # final downsampling (stride=1)
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        sequence += [
            nn.Conv2d(in_channels=ndf * nf_mult_prev,
                      out_channels=ndf * nf_mult,
                      kernel_size=kw,
                      stride=1,
                      padding=padw,
                      bias=use_bias),
            norm_layer(num_features=ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # Output layer (1-channel prediction map)
        sequence += [
            nn.Conv2d(in_channels=ndf * nf_mult,
                      out_channels=1,
                      kernel_size=kw,
                      stride=1,
                      padding=padw)
        ]

        self.main = nn.Sequential(*sequence)



    def forward(self, input):

        return self.main(input)
    




if __name__ == "__main__":

    from torchsummary import summary

    disc = Discriminator(
        input_nc=3,
        ndf=64,
        n_layers=3,
        use_actnorm=True
    )

    input_image = torch.randn(4, 3, 256, 256)
    # output_bn = disc(input_image)
    # print(f"shape of Output: {output_bn.shape}")

    # summary(output_bn, (3, 256, 256), device="cpu") # Library error 

    # Compute gan 
    from gan import UNetGenerator
    generator = UNetGenerator(input_nc=3,
                              output_nc=3,
                              ngf=64,
                              norm_layer=nn.BatchNorm2d)
    
    
    fake_image = generator(input_image)
    real_image = disc(fake_image)

    print(f"Generator output shape: {fake_image.shape}")
    print(f"Discriminator output shape: {real_image.shape}")

    fake_prob = torch.sigmoid(real_image)
    
    # check statistics 
    mean_prob = fake_prob.mean().item() 
    print(f"Discriminator avgerage `real` prob of `fake`: {mean_prob:.3f}")

    # mean_prob ~ 0.0    --> fake 
    # mean_prob > 0.5   --> real image [fool]
    # mean_prob ~ 0.5   -> uncertain 
