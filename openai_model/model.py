import torch
from torch import nn 
import math
from openai_model.utils import (
    conv_nd, 
    avg_pool_nd, 
    normalization, 
    linear, 
    zero_module,
    checkpoint,
    timestep_embedding
)

from openai_model.attention import FlashAttention
from torch.nn import functional as F
from abc import abstractmethod

from openai_model.attention import SpatialTransformer, AttentionPool2d, AttentionBlock



class TimestepBlock(nn.Module):

    """ 
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """ 
        Apply the module to `x` given `emb` timestep embeddings.
        """




class TimestepEmbedSequential(nn.Sequential,
                              TimestepBlock):
    
    """ 
    A sequential module that passes timestep embeddings to the children that 
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):

        # print("what type of data get [TimestepEmbedSequential] data : -->", x.shape)
        # print("what type of data get [TimestepEmbedSequential] embedding : -->", emb.shape)
        # print("what type of data get [TimestepEmbedSequential] context : -->", context.shape)

        for layer in self:
            # print("Layer: ", layer) # Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            if isinstance(layer, TimestepBlock):
                # print("[TiemstepBlock] is working...")
                x = layer(x, emb)
                

            elif isinstance(layer, SpatialTransformer):
                # print("[SpatialTransformer] is working...")
                x = layer(x, context)
                # print("what is the output [After spatial Transformer]: ->", x.shape)

            else:
                # print("[TimestepEmbeddingSequential] Else block is working...")
                x = layer(x)

        return x 



class Downsample(nn.Module):

    """ 
    A downsampling layer with an optional conv.
    :param channels: channels in the inputs and outputs
    :param use_conv: a bool determining if a conv is applied.
    :param dims: determines if the signal is 1D, 2D or 3D, if 3D then 
                    downsampling occures in the inner-two dimension.
    """

    def __init__(self,
                 channels,
                 use_conv,
                 dims=2,
                 out_channels=None,
                 padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims 
        stride = 2 if dims != 3 else (1, 2, 2)

        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )

        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)


    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)
    

class Upsample(nn.Module):

    """ 
    An upsampling layer with an optional conv.
    :param channels: channels in the inputs and output.
    :param use_conv: a bool determining if a conv is applied.
    :param dims: determines if the signal is 1D, 2D or 3D, If 3D , then
                    upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, 
                 channels, 
                 use_conv,
                 dims=2,
                 out_channels=None,
                 padding=1):
        
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)


    def forward(self, x):

        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                input=x, 
                size = (x.shape[2],
                        x.shape[3] * 2,
                        x.shape[4] * 2),
                mode="nearest"
            )

        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")

        if self.use_conv:
            x = self.conv(x)

        return x 
    



    
# class ResnetBlock(TimestepBlock):

#     """ 
#     A residual block that can optionally change the number of channels.
#     :param channels: the number of input channels
#     :param emb_channels: the number of timestep embedding channels 
#     :param output: the rate of dropout.
#     :param out_channels: if specified, the number of out channels.
#     :param use_conv: if True and out_channels is specified, use a spatial 
#                         conv instead of a similar 1x1 conv to change the 
#                         channels in the skip connection.
#     :param dims: determines if the signal is 1D, 2D or 3D 
#     :param use_checkpoint: if True, use gradient checkpointing on this module.
#     :param up: if True, use this block for upsampling 
#     :param down: If True, use this block for downsampling.
#     """

#     def __init__(
#             self, 
#             channels,
#             emb_channels,
#             dropout,
#             out_channels=None,
#             use_conv=False,
#             use_scale_shift_norm=False,
#             dims=2,
#             use_checkpoint=False,
#             up=False,
#             down=False
#     ):
        
#         super().__init__()
#         self.channels = channels
#         self.emb_channels = emb_channels
#         self.dropout = dropout
#         self.out_channels = out_channels or channels
#         self.use_conv = use_conv
#         self.use_checkpoint = use_checkpoint
#         self.use_scale_shift_norm = use_scale_shift_norm

#         self.in_layers = nn.Sequential(
#             normalization(channels),
#             nn.SiLU(),
#             conv_nd(dims, channels, self.out_channels, 3, padding=1)
#         )

#         self.updown = up or down 

#         if up:
#             self.h_upd = Upsample(channels=channels, use_conv=False, dims=dims)
#             self.x_upd = Upsample(channels=channels, use_conv=False, dims=dims)

#         elif down:
#             self.h_upd = Downsample(channels=channels, use_conv=False, dims=dims)
#             self.x_upd = Downsample(channels=channels, use_conv=False, dims=dims)

#         else:
#             self.h_upd = self.x_upd = nn.Identity()


#         self.emb_layers = nn.Sequential(
#             nn.SiLU(),
#             linear(emb_channels,
#                    2 * self.out_channels if use_scale_shift_norm else self.out_channels)
#         )

#         self.out_layers = nn.Sequential(
#             normalization(self.out_channels),
#             nn.SiLU(),
#             nn.Dropout(p=dropout),
#             zero_module(
#                 conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
#             )
#         )

#         if self.out_channels == channels:
#             self.skip_connection = nn.Identity()

#         elif use_conv:
#             self.skip_connection = conv_nd(
#                 dims, channels, self.out_channels, 3, padding=1
#             )

#         else:
#             self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)



#     def forward(self, x, emb):

#         """ 
#         Apply the block to a Tensor, conditioned in a timestep embedding.
#         :param x: an [N x C x ...] Tensor of feature.
#         :param emb: an [N x emb_channels] Tensor of timestep embedding.
#         :return an [N x C x ...] Tensor of outputs.
#         """

#         return checkpoint(
#             func=self._forward, 
#             inputs=(x, emb),
#             params=self.parameters(),
#             flag=self.use_checkpoint
#         )
    

#     def _forward(self, x, emb):

#         if self.updown:
#             in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
#             h = in_rest(x)
#             h = self.h_upd(h)
#             h = self.x_upd(x)
#             h = in_conv(h)


#         else:
#             # print("is this working....")
#             # print("what is the input data: ", x)
#             h = self.in_layers(x)

#         emb_out = self.emb_layers(emb).type(h.dtype)

#         while len(emb_out.shape) < len(h.shape):
#             emb_out = emb_out[..., None]

#         if self.use_scale_shift_norm:
#             out_norm = out_rest = self.out_layers[0], self.out_layers[1:]
#             scale, shift = torch.chunk(emb_out, 2, dim=1)

#             h = out_norm(h) * (1 + scale) + shift
#             h = out_rest(h)

#         else:
#             h = h + emb_out
#             h = self.out_layers(h)

#         return self.skip_connection(x) + h 

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class UnetModel(nn.Module):

    """ 
    The Full Unet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor 
    :param model_channels: base channel count for the model 
    :param out_channels: channels int the output tensor
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolution: a collection of downsample rates at which
        attention will take place. May be a set, list or tuple.
        for example, if this contains 4, then at 4x downsample attention will be used.
    
    :param dropout: the dropout probability 
    :param channel_mult: channel multiplier for each level of the Unet.
    :param conv_resample: if True, use learned conv for upsampling and downsampling.
    :param dims: determines if the signal is 1D, 2D or 3D 
    :param num_classes: if specified (as an int), then this model will be 
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use 
                            a fixed channel with per attention head.
    :param num_heads_upsample: works with num_heads to set a different number of
                            head for upsampling. Deprecated.
    :param use_scale_shift_norm: use a Film-like conditioning machanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially 
                            increased efficiency.
    """

    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=True,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,
            transformer_depth=1,
            context_dim=None,
            n_embed=None,
            legacy=True
    ):
        
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! you forgot to use the spatial transformer for your cross-attention conditioning...'

            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, "Either num_heads or num_head_channels has to be set"


        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolution = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None


        time_embed_dim = model_channels * 4 
        # print("what is the data of model_channels: ", model_channels)
        # print("what is the data of time_embed_dim: ", time_embed_dim)
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim)
        )
        # print("check the dtype of [After] time_emb: ", self.time_embed)


        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        
# <---------------------------------------------------- Middle Block ----------------------------------------------------------------->
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm

            ),
            AttentionBlock(
                channels=ch,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_checkpoint=use_checkpoint,
                use_new_attention_order=use_new_attention_order
            ) if not use_spatial_transformer else SpatialTransformer(
                in_channels=ch,
                n_heads=num_heads,
                d_head=dim_head,
                depth=transformer_depth,
                context_dim=context_dim
            ),
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            )
        )
        self._feature_size += ch 

        # <-------------------------------------------------------------------------- Output block ------------------------------------------------------------>
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                normalization(ch),
                conv_nd(dims, model_channels, n_embed, 1)
            )

    def convert_to_fp16(self):

        """Convert the tensor of the model to float16"""

        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)


    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):

        """ 
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of input.
        :param timestep: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        assert (y is not None) == (self.num_classes is not None), "must be specifiy [y] and [num_classes] is not none."

        hs = []
        

        # torch.Size([4]) -> torch.Size([4, 64])
        t_emb = timestep_embedding(timesteps=timesteps,
                                   dim=self.model_channels,
                                   repeat_only=False)
        
        # torch.Size([4, 64]) -> torch.Size([4, 256])
        # print("Let's check the dtype of t_emb: ", t_emb)
        # print("Let's check the dtype of t_emb: ", t_emb.dtype)
        emb = self.time_embed(t_emb.half())
        
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],), "must be equal to the shape of [y] and [x]"

            # torch.Size([4, 256]) +  torch.Size([4, 256]) -> torch.Size([4, 256])
            emb = emb + self.label_emb(y)

        # check the dtype of data
        h = x.type(self.dtype)
        # print("check the data type of h:", h.dtype)

        for module in self.input_blocks:
            
            # torch.Size([4, 64, 64, 64]), torch.Size([4, 256]), torch.Size([4, 77, 512]) -> torch.Size([4, 64, 64, 64])
            h = module(h, emb, context)
            hs.append(h)

        # torch.Size([4, 64, 64, 64]), torch.Size([4, 256]), torch.Size([4, 77, 512]) -> torch.Size([4, 512, 8, 8])
        h = self.middle_block(h, emb, context)
        # print("what is the shape [After middle block]: ", h.shape)

        for module in self.output_blocks:
            skip = hs.pop()
            h = torch.cat([h, skip], dim=1)
           


            


            


def convert_module_to_f16(x):
    x.dtype()

        




if __name__ == "__main__":

    unet = UnetModel(
        image_size=64,
        in_channels=3,
        out_channels=3,
        model_channels=224,
        attention_resolutions=[8, 4, 2],
        num_res_blocks=2,
        channel_mult=(1, 2, 3, 4),
        num_head_channels=32,
        
        num_classes=10,
    ).half().cuda()

    total_param = sum(p.numel() for p in unet.parameters())
    print(f"Model parameters: {total_param/1e6:.2f}M")

    
    # create sample inputs 
    batch_size = 4 
    x = torch.randn(batch_size, 3, 64, 64).half().cuda()
    timesteps = torch.tensor([500] * batch_size).cuda()
    context = torch.randn(batch_size, 77, 512).half().cuda()
    y = torch.randint(0, 10, (batch_size,)).cuda()


    # Forward padd 
    with torch.no_grad():
        output = unet(x, timesteps, context, y) 

    # print(output)