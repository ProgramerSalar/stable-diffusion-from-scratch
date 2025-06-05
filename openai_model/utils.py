import torch 
from torch import nn 
import math 
from einops import repeat


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        
        if x.device.type == 'cpu' and x.dtype == torch.float16:
            with torch.autocast('cpu', enabled=False):
                return super().forward(x.float()).type(x.dtype)
        
        return super().forward(x)



def conv_nd(dims, *args, **kwargs):

    """ Create a 1D, 2D conv modules. """

    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    
    raise ValueError(f"Unsupported dims: {dims}")


def avg_pool_nd(dims, *args, **kwargs):

    """ Create a 1D, 2D or 3D average pooling module."""

    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class CheckpointFunction(torch.autograd.Function):

    """
    Custom autograd function for gradient checkpointing.
    Reduce memory usage by tracking compute for memory during backward pass.
    """

    @staticmethod
    def forward(ctx, run_function, length, *args):

        """ 
        Forward pass with checkpointing.
        Args:
            ctx: context object to store information for backward pass 
            run_function: Function to execute during forward/backward 
            length: Number of input tensors in *args 
            *args: Arguments for run_function (tensors * parameters)
        """

        # store the function and arguments in context
        ctx.run_function = run_function
        # split args: first 'length' items are input tensors
        ctx.input_tensors = list(args[:length])
        # Remaining items are parameters (weights, biases)
        ctx.input_params = list(args[length:])

        # Execute without tracking gradients 
        with torch.no_grad():
            # IMPORTANT: Pass both inputs AND parameters to function
            output_tensors = ctx.run_function(*ctx.input_tensors)

        return output_tensors
    

    @staticmethod
    def backward(ctx, *output_grads):

        # prepare input tensors for recomputation.
        # detach(): Remove from computation graph
        # .requires_grad_(True): Enables gradient tracking
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        
        # Re-enableds gradient calculation for recomputation.
        with torch.enable_grad():
            # create shallow copies of tensors (avoids in-place modification issues)
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            # Recompute forward pass using stored function and modified inputs.
            output_tensors = ctx.num_function(*shallow_copies)


        # compute gradients via autograd
        input_grads = torch.autograd.grad(
            outputs=output_tensors,     # Recomputed tensors
            inputs=ctx.input_tensors + ctx.input_tensors,   # input tensors + parameters
            grad_outputs=output_grads,  # output gradients
            allow_unused=True   # ignore inputs without gradients
        )

        # deletes store references to free memory
        del ctx.input_tensors 
        del ctx.input_params 
        del output_tensors

        # Return gradients
        # Two none for run_function and length (non-tensor arguments)
        # input_grads for input tensors and parameters
        return (None, None) + input_grads
    


        




def checkpoint(func, inputs, params, flag):

    """
    Evaluate a function without caching intermediate activation, allowing for 
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`
    :param params: a sequence of parameters `func` depends on but does not 
                    explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """

    

    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    
    else:
      
        a = func(*inputs)
       
        return func(*inputs)
    


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding