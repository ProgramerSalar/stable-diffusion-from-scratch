import torch, math
from torch import nn 
from .attention import make_attention




class Upsample(nn.Module):

    def __init__(self,
                 in_channels,
                 with_conv):
        
        super().__init__()
        self.with_conv = with_conv

        if self.with_conv:
            self.conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=in_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
            

    def forward(self, x):
        x = nn.functional.interpolate(input=x,
                                      scale_factor=2.0,
                                      mode="nearest")
        

        if self.with_conv:
            x = self.conv(x)

        return x 
    

class Downsample(nn.Module):

    def __init__(self,
                 in_channels,
                 with_conv):
        
        super().__init__()
        self.with_conv = with_conv

        if self.with_conv:
            self.conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=in_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=0)
            

    def forward(self, x):

        if self.with_conv:
            pad = (0, 1, 0, 1)  # (left, right, top, bottom)

            x = nn.functional.pad(input=x, 
                                  pad=pad,
                                  mode="constant",
                                  value=0)
            
            x = self.conv(x)

        else:
            x = nn.functional.avg_pool2d(input=x,
                                         kernel_size=2,
                                         stride=2)
            
        return x 
    

def Normalize(in_channels, 
              num_groups=32):
    
    print("what is the input channels to the GroupNorm: ", in_channels)
    
    conv = nn.GroupNorm(num_groups=num_groups,
                        num_channels=in_channels,
                        eps=1e-6,
                        affine=True)

    return conv 



def nonlinearity(x):
    return x * torch.sigmoid(x)




class ResnetBlock(nn.Module):

    def __init__(self,
                 *,
                 in_channels, 
                 out_channels=None,
                 conv_shortcut=False,
                 dropout,
                 temb_channels=512):
        


        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               padding=1,
                               stride=1)
        
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        

        if temb_channels > 0:
            self.temb_proj = nn.Linear(in_features=temb_channels,
                                       out_features=out_channels)
            

        if self.in_channels != self.out_channels:

            if self.use_conv_shortcut:
                self.conv_sortcut = nn.Conv2d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1)
                

            else:
                self.nin_shortcut = nn.Conv2d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0)



    def forward(self, x, temb):

        h = x 
        # print("what is the shape of input normalization: ", h.shape)
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            # shape of h = [Batch_size, channels, height, width], shape of temb = [Batch_size, channels] + [None, None]

            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_sortcut(x)

            else:
                x = self.nin_shortcut(x)

        return x + h 
    




        

        
class Model(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 double_z=True,
                 resolution, use_timestep=True, use_linear_attn=False, attn_type="vanilla"):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attention(block_in, attention_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attention(block_in, attention_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        

        

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attention(block_in, attention_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t=None, context=None):
        #assert x.shape[2] == x.shape[3] == self.resolution
        if context is not None:
            # assume aligned context, cat along channel axis
            x = torch.cat((x, context), dim=1)
        if self.use_timestep:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        print("hs: ", hs)

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
        




def get_timestep_embedding(timesteps, embedding_dim):

    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2 
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)

    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1, 0, 0))

    return emb 









if __name__ == "__main__":

    # x = torch.randn(1, 64, 32, 32)
    # temb = torch.randn(1, 512)


    # resnet_block = ResnetBlock(in_channels=64,
    #                            out_channels=128, 
    #                            conv_shortcut=True,
    #                            dropout=0.1)
    
    # print(resnet_block)

    # ----------------------------------------------------------------------------

    import os 

    # Enable expandable segments to reduce fragmentation 
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.cuda.empty_cache()

    # Use float16 if possible 
    torch.backends.cudnn.benchmark = True

    x = torch.randn(1, 3, 256, 256)
    t = torch.tensor([1000])



    model = Model(ch=128,
                  out_ch=3,
                  ch_mult=(1, 2, 4, 4),
                  attn_resolutions=[],
                  resolution=256,
                  num_res_blocks=1,
                  in_channels=3
                  ).to('cpu')
    # print(model)

    # Run with mix-precision 
    with torch.cuda.amp.autocast():
        output = model(x.to('cpu'), t.to('cpu'))
        output = output.to("cuda")
    print(output.shape)

    
