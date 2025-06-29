import torch 
from torch import nn, einsum 
from inspect import isfunction
from einops import rearrange, repeat
from openai_model.utils import checkpoint, conv_nd, normalization
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
import math
import numpy as np 

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d




class CrossAttention(nn.Module):

    def __init__(self,
                 query_dim,
                 context_dim=None,
                 heads=8,
                 dim_head=64,
                 dropout=0.):
        
        super().__init__()
        inner_dim = dim_head * heads 
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5 
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        self.convert_to_float16()

    def convert_to_float16(self):

        """Convert all linear layer parameters to float16"""
        for layer in [self.to_q, self.to_k, self.to_v, self.to_out[0]]:
            layer.weight.data = layer.weight.data.half()

            if layer.bias is not None:
                layer.bias.data = layer.bias.data.half()

                


    def forward(self, x, context=None, mask=None):
        print(f"what is the dtype of functon crossAttention : >>> {x.dtype}")
        h = self.heads 
        context = default(context, x)

        print(f"what is the dtype of query: >>>> {self.to_q}")
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        

        # prepare mask for flash attention 
        batch_size, seq_len_q = q.shape[0], q.shape[1]
        if mask is not None:
            # [batch_size, seq_len, dim_head] -> [batch_size * heads, seq_len, dim] 
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
            
            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
            
            # Prepare mask
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask.bool(), max_neg_value)
            
            # Attention and output
            attn = sim.softmax(dim=-1)
            out = torch.einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

            return self.to_out(out)

        


        # Reshape for flash attention: [batch_size, seq_len, heads, dim_head]
        q = rearrange(q, 'b n (h d) -> b n h d', h=h)
        k = rearrange(k, 'b n (h d) -> b n h d', h=h)
        v = rearrange(v, 'b n (h d) -> b n h d', h=h)


        # Flash attention 
        out = flash_attn_func(
            q, k, v,
            dropout_p=0.0,
            softmax_scale=self.scale,   # same scaling as original
            causal=False,   # non-autoregressive
            
        )

        # reshape 
        out = rearrange(out, 'b s h d -> b s (h d)')

        return self.to_out(out)

        
    








class GELU(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)


    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * nn.functional.gelu(gate)




class FeedForward(nn.Module):

    def __init__(self, 
                 dim,
                 dim_out=None,
                 mult=4,
                 glu=False,
                 dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GELU(dim, inner_dim)


        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )


    def forward(self, x):
        return self.net(x)
    













class BasicTransformerBlock(nn.Module):

    def __init__(self,
                 dim, 
                 n_heads,
                 d_head,
                 dropout=0.,
                 context_dim=None,
                 gated_ff=True,
                 checkpoint=True):
        
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim,
                                    heads=n_heads,
                                    dim_head=d_head,
                                    dropout=dropout)
        self.ff = FeedForward(dim=dim,
                              dropout=dropout,
                              glu=gated_ff)
        
        self.attn2 = CrossAttention(query_dim=dim,
                                    context_dim=context_dim,
                                    heads=n_heads,
                                    dim_head=d_head,
                                    dropout=dropout
                                    )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint


    def forward(self, x, context=None):
        return checkpoint(func=self._forward,
                          inputs=(x, context),
                          params=self.parameters(),
                          flag=self.checkpoint)
    

    def _forward(self, x, context=None):
        # print(f"check the data type : {x}")
        x_norm = x.to(torch.float32)
        print(f"check the data type : {x.shape} and dtype: >> {x.dtype}")   # torch.Size([4, 1024, 320]) and dtype: >> torch.float16

        
        norm1_x = self.norm1(x_norm)
        norm1_x = norm1_x.half()
        print(f"check the dtype of norm_x : {norm1_x.shape} and type: >>> {norm1_x.dtype}") # torch.Size([4, 1024, 320]) and type: >>> torch.float16
        x = self.attn1(norm1_x) + x

        x = x.to(torch.float32)
        norm2_x = self.norm2(x)
        norm2_x = norm2_x.half()
        x = self.attn2(norm2_x, context=context) + x

        # x = self.attn2(self.norm2(x), context=context) + x 
        x = self.ff(self.norm3(x)) + x 

        return x 
    

def zero_module(module):

    """ 
    zero output the parameters of a module and return it.
    """

    for p in module.parameters():
        p.detach().zero_()

    return module



    

class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):

        # print("what is the input data in [spatialTransformer]: ", x.shape)
        # print("what is the context in [spatialTransformer]: ", context.shape)
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        #  torch.Size([4, 512, 8, 8]) -> torch.Size([4, 512, 8, 8])
        x = self.norm(x)
        # torch.Size([4, 512, 8, 8]) -> torch.Size([4, 512, 8, 8])
        x = self.proj_in(x)
        # torch.Size([4, 512, 8, 8]) ->  torch.Size([4, 64, 512])
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        for block in self.transformer_blocks:
            # print("block: ", block)
            x = block(x, context=context)
            # print("what is the shape [After Transformer Block]: ", x.shape)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)

        output = x + x_in
        # print("what is the output of [spatialTransformer]: ", output.shape)
        return x + x_in
    




class FlashAttention(nn.Module):

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        
        bs, width, seq_len = qkv.shape 
        assert width % (3 * self.n_heads) == 0 
        head_dim = width // (3 * self.n_heads)

        
        # [bs, width, seq_len] -> [bs, seq_len, width]
        qkv = qkv.permute(0, 2, 1)
        # [bs, seq_len, width] -> [bs, seq_len, 3, n_heads, head_dim]
        qkv = qkv.view(bs, seq_len, 3, self.n_heads, head_dim)
        
        
        # [bs, seq_len, 3, n_heads, head_dim] -> [bs, seq_len, n_heads, head_dim]
        out = flash_attn_qkvpacked_func(
            qkv,
            dropout_p=0.0,
            softmax_scale=1.0/math.sqrt(head_dim),
            causal=False
        )
        
    
        # [bs, seq_len, n_heads, head_dim] -> [bs, n_heads, seq_len, head_dim]
        out = out.permute(0, 2, 1, 3)
        # [bs, n_heads, seq_len, head_dim] -> [bs, seq_len, n_heads*head_dim]
        out = out.reshape(bs, seq_len, self.n_heads * head_dim)
        # [bs, seq_len, n_heads*head_dim] -> [bs, n_heads*head_dim, seq_len]
        out = out.permute(0, 2, 1)

        return out 
    

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model=model,
                                _x = _x,
                                y=y)
    

def count_flops_attn(model, _x, y):

    """ 
    A counter for the `thop` package to counter the operations in an 
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timesteps),
            custom_op={QKVAttention: QKVAttention.count_flops}
            )
    """

    b, c, *spatial = y[0].shape 
    num_spatial = int(np.prod(spatial))

    # We perform two matmuls with the same number of operation.
    # The first computes the weight matrix, the second computes 
    # the combines of the value vactors.

    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])



class AttentionPool2d(nn.Module):

    def __init__(
            self,
            spacial_dim:int,    # spatial dim (height/width) of input feature map
            embed_dim: int,
            num_heads_channels: int,
            output_dim: int = None  
    ):
        
        super().__init__()

        # 1. Positional embedding parameter
        # shape: [embed_dim, spatial_dim ** 2 + 1] 

        self.positional_embedding = nn.Parameter(torch.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = FlashAttention(self.num_heads)


    def forward(self, x):

        # [bs, channels, height, width] -> [bs, channels, spatial_dim]
        b, c, *_spatial = x.shape 
        # [bs, channels, spatial_channels] -> [bs, channels, Flatten_the_spatial_dim]
        x = x.reshape(b, c, -1) 
        
        # [bs, channels, Flatten_the_spatial_dim] -> [bs, channels, 1]
        global_token = x.mean(dim=-1, keepdim=True)
        
        # [bs, channels, Flatten_the_spatial_dim] + [bs, channels, 1] -> [bs, channels, Flatten_the_spatial_dim]
        x = torch.cat([global_token, x], dim=-1)
        
        

        # [embed_dim, spatial_dim] -> [1, embed_dim, spatial_dim]
        positional_embedding = self.positional_embedding[None, :, :]
        
        # [bs, channels, Flatten_the_spatial_dim] + [1, embed_dim, spatial_dim] -> [bs, channels, Flatten_the_spatial_dim]
        x = x + positional_embedding.to(x.dtype)

        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
       

        # [bs, channels, Flatten_the_spatial_dim] -> [bs, channels]
        return x[:, :, 0]   
    
class QKVAttentionLegacy(nn.Module):

    """A module performs QKV attention. Matches legacy QKVAttention + input/output heads shaping"""

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):

        """ 
        Apply QKV attention.
        :param qkv: an [ N x (H * 3 * C) x T] tensor of Qs, Ks, Vs
        :return: an [N x (H * C) x T] tensor after attention
        """

        bs, width, length = qkv.shape 
        assert width % (3 * self.n_heads) == 0 
        ch = width // (3 * self.n_heads)

        # Reshape for flash attention: [N, T, 3, H, C]
        qkv = qkv.permute(0, 2, 1)
        qkv = qkv.view(bs, length, 3, self.n_heads, ch)

        scale = 1 / math.sqrt(math.sqrt(ch))

        output = flash_attn_qkvpacked_func(
            qkv,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=False
        )

        output = output.reshape(bs, length, -1)
        output = output.permute(0, 2, 1)

        return output
    

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model=model,
                                _x=_x,
                                y=y)
    




class AttentionBlock(nn.Module):

    """ 
    An attention block that allows spatial positions to attend to each other.
    """


    def __init__(self,
                 channels,
                 num_heads=1,
                 num_head_channels=-1,
                 use_checkpoint=False,
                 use_new_attention_order=False):
        
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads

        else:
            assert (
                channels % num_head_channels == 0
            ), f"q, k, v channels {channels} is not divisible by num_head_channels {num_head_channels}"

            self.num_heads = channels // num_head_channels


        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)

        if use_new_attention_order:
            # split qkv before split heads 
            self.attention = FlashAttention(n_heads=self.num_heads)

        else:
            # split heads before split qkv 
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))


    def forward(self, x):

        return checkpoint(self._forward, (x,), self.parameters(), True) # TODO: check checkpoint usage, is True # TODO: Fix the .half call 
        

    def _forward(self, x):

        b, c, *spatial = x.shape 
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
         
        output = (x + h).reshape(b, c, *spatial)
        # print("output: ", output.shape)

        return output
    




if __name__ == "__main__":

    # attn = CrossAttention(
    #     query_dim=256,  # Dim of query vector 
    #     context_dim=512,    # Dim of context vector 
    #     heads=8,
    #     dim_head=64,
    #     dropout=0.1
    # ).half().cuda()

    # batch_size = 4
    # query_len = 10 
    # context_len = 15 

    # # query tensor: [bs, query_len, query_dim]
    # x = torch.randn(batch_size, query_len, 256).half().cuda()

    # # context tensor: [bs, context_len, context_dim]
    # context = torch.randn(batch_size, context_len, 512).half().cuda()

    # mask = torch.ones(batch_size, context_len, dtype=torch.bool).cuda()
    # mask[:, -5:] = False 

    # print(mask.shape)
    # print(mask)

    # output = attn(x, context, mask=mask)   # # (4, 10, 256)
    # print(output.shape)
    # print(output)
    # ----------------------------------------------------------------------

    # batch_size = 2 
    # num_heads = 4 
    # channels_per_head = 8
    # seq_len = 16 

    # total_channels = 3 * num_heads * channels_per_head  # 144
    # qkv = torch.randn(batch_size, total_channels, seq_len).half().cuda() # (2, 144, 16)

    # # output = QKVAttention(n_heads=num_heads)
    # output = FlashAttention(n_heads=num_heads).half().cuda()
    # output = output(qkv)
    # print(output.shape) # (2, 48, 16)
    # ------------------------------------------------------------------------------------------------------

    # pool = AttentionPool2d(
    #     spacial_dim=32,
    #     embed_dim=256,
    #     num_heads_channels=64,
    #     output_dim=512
    # ).half().cuda()

    # x = torch.randn(2, 256, 32, 32).half().cuda()

    # output = pool(x)
    # print(output.shape)

    # -----------------------------------------------------------------
    # batch_size = 2
    # num_heads= 4 
    # channels_per_head= 8
    # seq_len = 16

    # qktattention_legacy = QKVAttentionLegacy(n_heads=num_heads).half().cuda()
    

    # total_channels = 3 * num_heads * channels_per_head # 96
    # print(total_channels)
    # qkv = torch.randn(batch_size, total_channels, seq_len).half().cuda() # (2, 96, 16)
    # print(qkv.shape)

    # output = qktattention_legacy(qkv)
    # print(output.shape)

    # --------------------------------------------------------------------------------------------------

    attention_block = AttentionBlock(channels=128,
                                     num_heads=4,
                                     num_head_channels=-1,
                                     use_checkpoint=True).half().cuda()
    
    x = torch.randn(4, 128, 62).half().cuda()
    # print("x: ", x)
    output = attention_block(x)
    output