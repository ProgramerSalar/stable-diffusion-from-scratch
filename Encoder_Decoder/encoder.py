import torch 
from torch import nn 
from Unet.unet import ResnetBlock, Downsample, Normalize, nonlinearity, Upsample
from Unet.attention import make_attention
import numpy as np 

class Encoder(nn.Module):

    def __init__(self,
                 *,
                 ch,
                 out_ch,
                 ch_mult=(1, 2, 4, 8),
                 num_res_blocks,
                 attn_resolutions,
                 dropout=0.0,
                 resamp_with_conv=True,
                 in_channels,
                 resolution,
                 z_channels,
                 double_z=False,
                 use_linear_attn=False,
                 attn_type="vanilla",
                 **ignore_kwargs
                 ):
        

        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch 
        self.temb_ch = 0 
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling 
        self.conv_in = torch.nn.Conv2d(in_channels=in_channels, out_channels=self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))

                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attention(in_channels=block_in, attention_type=attn_type))

            down = nn.Module()
            down.block = block
            down.attn = attn

            if i_level != self.num_resolutions -1:
                down.downsample = Downsample(in_channels=block_in, with_conv=resamp_with_conv)
                curr_res = curr_res // 2 

            self.down.append(down)

        
        # middle 
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = make_attention(in_channels=block_in, attention_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        # ENd 
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(in_channels=block_in, out_channels=2*z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):

        # Timestep embedding 
        temb = None 

        # downsampling 
        hs = [self.conv_in(x)]
        print("shape of hs: ", hs[-1].shape)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):

                h = self.down[i_level].block[i_block](hs[-1], temb)
                print("shape of h: ", h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)

                hs.append(h)

            if i_level != self.num_resolutions -1:
                hs.append(self.down[i_level].downsample(hs[-1]))


        # Middle 
        hs = hs[-1]
        h = self.mid.block_1(h, temb)
        print("shape of block_1: ", h.shape)
        h = self.mid.attn_1(h)
        print("shape of attention_1: ", h.shape)
        h = self.mid.block_2(h, temb)
        print("shape of block_2: ", h.shape)

        # End 
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        print("shape of out_conv: ", h.shape)

        return h 



class Decoder(nn.Module):

    def __init__(self,
                 *,
                 ch,
                 out_ch,
                 ch_mult=(1, 2, 4, 8),
                 num_res_blocks,
                 attn_resolutions,
                 dropout=0.0,
                 resamp_with_conv=True,
                 in_channels,
                 resolution,
                 z_channels,
                 give_pre_end=False,
                 tanh_out=False,
                 use_linear_attn=False,
                 attn_type="linear",
                 **ignorekwargs):
        
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch 
        self.temb_ch = 0 
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res 
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions -1]
        curr_res = resolution // 2 ** (self.num_resolutions -1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print(f"Working with z of shape {self.z_shape} = {np.prod(self.z_shape)} dimensions")


        # z to block_in 
        self.conv_in = nn.Conv2d(in_channels=z_channels, out_channels=block_in, kernel_size=3, stride=1, padding=1)

        # middle 
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = make_attention(in_channels=block_in, attention_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        # upsampling 
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))

                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attention(in_channels=block_in, attention_type=attn_type))

            up = nn.Module()
            up.block = block 
            up.attn = attn 

            if i_level != 0:
                up.upsample = Upsample(in_channels=block_in, with_conv=resamp_with_conv)
                curr_res = curr_res * 2 

            self.up.insert(0, up)



        # End 
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(in_channels=block_in, out_channels=out_ch, kernel_size=3, stride=1, padding=1)


    
    def forward(self, z):

        self.last_z_shape = z.shape 

        # timestep embedding 
        temb = None 

        # z to block_in 
        h = self.conv_in(z)

        # middle 
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling 
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks +1):
                
                h = self.up[i_level].block[i_block](h, temb)

                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

            if i_level != 0:
                h = self.up[i_level].upsample(h)


        # End 
        if self.give_pre_end:
            return h 
        
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        if self.tanh_out:
            h = torch.tanh(h)

        return h 
    

def to_float8(tensor):
    return tensor.half()

if __name__ == "__main__":

    ddconfig = {
        "ch": 128,
        "out_ch": 3,
        "ch_mult": (1, 1, 2, 2, 4),
        "num_res_blocks": 1,
        "attn_resolution": [16],
        "in_channels": 3,
        "resolution": 256,
        "double_z": True,
        "z_channels": 16
    }

    import os 

    # Enable expandable segments to reduce fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.cuda.empty_cache()

    # Use FP16 if possible
    torch.backends.cudnn.benchmark = True

    x = torch.randn(1, 3, 256, 256)

    encoder = Encoder(ch=ddconfig["ch"],
                      out_ch=ddconfig["out_ch"],
                      ch_mult=ddconfig["ch_mult"],
                      num_res_blocks=ddconfig["num_res_blocks"],
                      attn_resolutions=ddconfig["attn_resolution"],
                      in_channels=ddconfig["in_channels"],
                      resolution=ddconfig["resolution"],
                      double_z=ddconfig["double_z"],
                      z_channels=ddconfig["z_channels"]).to("cuda")
    
    
    # print(encoder)

    
    # Run with mixed precision 
    with torch.cuda.amp.autocast():
        z = encoder(x.to("cuda"))
        z = z.to("cuda")
    print(f"{z} what is the shape of z: {z.shape}")

    # -------------------------------------------------------------------------------------
    # # Enable expandable segments to reduce fragmentation
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # torch.cuda.empty_cache()


    # # Use float16 if possible 
    # torch.backends.cudnn.benchmark = True

    # decoder = Decoder(ddconfig)

    # # Freeze model if not training 
    # decoder.eval()
    
    # print(decoder)

    # # Momery efficient forward pass 
    # with torch.no_grad():
    #     output = decoder(z)


    # print(output.shape)
