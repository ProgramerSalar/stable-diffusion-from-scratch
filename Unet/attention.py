import torch 
from torch import nn 
from einops import rearrange


class LinearAttention(nn.Module):

    def __init__(self, 
                 dim, 
                 heads=4,
                 dim_head=32):
        
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(in_channels=dim,
                                out_channels=hidden_dim * 3,
                                kernel_size=1,
                                bias=False)
        
        self.to_out = nn.Conv2d(in_channels=hidden_dim,
                                out_channels=dim,
                                kernel_size=1)
        

    def forward(self, x):

        b, c, h, w = x.shape  # torch.Size([1, 64, 32, 32])
        qkv = self.to_qkv(x)  # torch.Size([1, 384, 32, 32])

        # [b, (3 * heads * c), h, w] -> [3, b, heads, c (h w)] => (1, 4, 32, 1024)
        q, k, v = rearrange(qkv, 
                            "b (qkv heads c) h w -> qkv b heads c (h w)",
                            heads=self.heads,
                            qkv=3)  # torch.Size([1, 4, 32, 1024])
        
        
        
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn, bhen -> bhde", k, v)
        out = torch.einsum("bhde, bhdn -> bhen", context, q)
        

        out = rearrange(out,
                        "b heads c (h w) -> b (heads c) h w",
                        heads=self.heads,
                        h=h,
                        w=w)
        
        out = self.to_out(out)
        
        
        return out
    

if __name__ == "__main__":

    x = torch.randn(1, 64, 32, 32)
    linear_attention = LinearAttention(dim=64)
    # print(linear_attention)

    output = linear_attention(x)
    print(output)