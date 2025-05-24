import torch 
from torch import nn 
# from Unet.unet import Normalize

# import unet
# from .unet import Normalize

from einops import rearrange

def Normalize(in_channels, 
              num_groups=32):
    
    conv = nn.GroupNorm(num_channels=in_channels, 
                        num_groups=num_groups,
                        eps=1e-6,
                        affine=True)

    return conv 


# class AttentionBlock(nn.Module):

#     def __init__(self,
#                  in_channels):
        
#         super().__init__()
#         self.in_channels = in_channels

#         self.norm = Normalize(in_channels)
#         self.q = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
#         self.k = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
#         self.v = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
#         self.proj_out = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        
        
#     def forward(self, x):

#         h_ = x 
#         h_ = self.norm(h_)
#         q = self.q(h_)
#         k = self.k(h_)
#         v = self.v(h_)

#         ## compute the attention with key, query 
#         b, c, h, w = q.shape 
#         # [b, c, h, w] -> [b, c, h*w] -> [b, h*w, c]
#         q = q.reshape(b, c, h*w).permute(0, 2, 1)
#         # [b, c, h, w] -> [b, c, h*w]
#         k = q.reshape(b, c, h*w)

#         # [b, h*w, c] bmm [b, c, h*w] -> [b, h*w, h*w]
#         w_ = torch.bmm(q, k)
#         # w_ = q @ k

#         # scale the color channels and multiply the weight(w_)
#         w_ = w_ * (int(c) * (-0.5))
#         # apply the softmax 
#         w_ = nn.functional.softmax(w_, dim=2)

#         ## compute the attention with values
#         # [b, c, h, w] -> [b, c, h*w]
#         v = v.reshape(b, c, h*w)
#         # permute the weight [b, h*w, h*w] -> [b, h*w, h*w]
#         w_ = w_.permute(0, 2, 1)
        
#         # [b, c, h*w] bmm [b, h*w, h*w] -> [b, c, h*w]
#         output = torch.bmm(v, w_)

#         # [b, c, h*w] -> [b, c, h, w]
#         output = output.reshape(b, c, h, w)

#         output = self.proj_out(output)

#         return x + output

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


class LinearAttention(nn.Module):

    def __init__(self,
                 dim,
                 heads=4,
                 dim_head=32):
        
        super().__init__()
        self.heads = heads 
        hidden_dim = dim_head * heads # 128
        self.to_qkv = nn.Conv2d(in_channels=dim, out_channels=hidden_dim * 3, kernel_size=1, bias=False) # [64, 384(head , channels)]
        self.to_out = nn.Conv2d(in_channels=hidden_dim, out_channels=dim, kernel_size=1)    # [128, 64]


    def forward(self, x):

        b, c, h, w = x.shape 
        # torch.Size([1, 64, 32, 32]) -> torch.Size([1, 384, 32, 32])
        qkv = self.to_qkv(x)
        
        # torch.Size([1, 384, 32, 32]) = 393216 -->  torch.Size([1, 4, 32, 1024]) = 131072 * 3 =>  393216 
        # Understanding of [32 => 32*4*3 => 384]
        # [b, (1 * heads * c) h w ] -> [b, heads, c, (h w)]
        q, k, v = rearrange(qkv, 
                            "b (qkv heads c) h w -> qkv b heads c (h w)",
                            heads=self.heads,
                            qkv=3
                            )
        
        
        k = k.softmax(dim=-1)
        

        # bhdn -> [batch_size, heads, key_of_dim, seq_len] 
        # bhen -> [batch_size, heads, key_of_value, seq_len]
        # bhde -> [batch_size, heads, dim, seq_len]
        context = torch.einsum("bhdn, bhen -> bhde", k, v)
        
        out = torch.einsum("bhde, bhdn -> bhen", context, q)

        # torch.Size([1, 4, 32, 1024]) -> torch.Size([1, 128, 32, 32])
        out = rearrange(out, 
                        "b heads c (h w) -> b (heads c) h w",
                        heads=self.heads,
                        h=h,
                        w=w)
        
        
        out = self.to_out(out)
    
        return out
    


class LinearAttentionBlock(LinearAttention):

    """to match Attention usage"""

    def __init__(self,
                 in_channels):
        
        super().__init__(dim=in_channels, 
                         heads=1,
                         dim_head=in_channels)
        


def make_attention(in_channels,
                   attention_type="vanilla"):
    
    assert attention_type in ["vanilla", "linear", "none"], f"attention_type {attention_type} not found."

    if attention_type == "vanilla":
        return AttentionBlock(in_channels)
    
    elif attention_type == "linear":
        return LinearAttentionBlock(in_channels)
    
    else:
        return nn.Identity(in_channels)
    












# if __name__ == "__main__":
#     x = torch.randn(1, 64, 32, 32)
#     linear_attention = LinearAttention()
#     # print(linear_attention)

#     output = linear_attention(x)
#     # print(output)


