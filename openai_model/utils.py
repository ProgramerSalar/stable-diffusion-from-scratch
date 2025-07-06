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

# class CheckpointFunction(torch.autograd.Function):

#     """
#     Custom autograd function for gradient checkpointing.
#     Reduce memory usage by tracking compute for memory during backward pass.
#     """

#     @staticmethod
#     def forward(ctx, run_function, length, *args):

#         print(f"what is input to get functon: [checkpointFunction] >>>> {ctx}")

#         """ 
#         Forward pass with checkpointing.
#         Args:
#             ctx: context object to store information for backward pass 
#             run_function: Function to execute during forward/backward 
#             length: Number of input tensors in *args 
#             *args: Arguments for run_function (tensors * parameters)
#         """

#         # store the function and arguments in context
#         ctx.run_function = run_function
#         print(f"what is the meaning of ctx.run_function in class [checkpointFunction] >>> {ctx.run_function}")

#         # split args: first 'length' items are input tensors
#         ctx.input_tensors = list(args[:length])
#         print(f"what is ctx.input_tensors: {ctx.input_tensors}")
#         # Remaining items are parameters (weights, biases)
#         ctx.input_params = list(args[length:])

#         # Execute without tracking gradients 
#         with torch.no_grad():
#             # IMPORTANT: Pass both inputs AND parameters to function
#             output_tensors = ctx.run_function(*ctx.input_tensors).half()
#             print(f"what is the output tensor of [class - CheckpointFunction] method - [forward]: >> {output_tensors} and dtype of this output tensor: >>> {output_tensors.dtype}")

#         return output_tensors
    

#     @staticmethod
#     def backward(ctx, *output_grads):

#         # prepare input tensors for recomputation.
#         # detach(): Remove from computation graph
#         # .requires_grad_(True): Enables gradient tracking
#         ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
#         print(f"what is the input tensor of function [checkpointFunction] method: [backward] >>> {ctx.input_tensors}")
        
#         # Re-enableds gradient calculation for recomputation.
#         with torch.enable_grad():
#             # create shallow copies of tensors (avoids in-place modification issues)
#             shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
#             print(f"what is the output to get to this shallow_copies: function [backward] class [CheckpointFunction]: >>>> {shallow_copies}")

#             # Recompute forward pass using stored function and modified inputs.
#             output_tensors = ctx.run_function(*shallow_copies)
#             print(f"what is the output tensor to [class - CheckpointFunction] method [backward] >>> {output_tensors} and shape = {output_tensors.shape}")


#         # compute gradients via autograd
#         input_grads = torch.autograd.grad(
#             outputs=output_tensors,     # Recomputed tensors
#             inputs=ctx.input_tensors + ctx.input_tensors,   # input tensors + parameters
#             grad_outputs=output_grads,  # output gradients
#             allow_unused=True   # ignore inputs without gradients
#         )

#         print(f"what is the input gradients in [class-CheckpointFunction]: >>> {input_grads} and what is the shape = {input_grads[0].shape}, {input_grads[1].shape}, {input_grads[2].shape}, {input_grads[3].shape}")
#         # torch.Size([4, 1024, 320]), torch.Size([4, 77, 768]), torch.Size([4, 1024, 320]), torch.Size([4, 77, 768])

        

#         # deletes store references to free memory
#         del ctx.input_tensors 
#         del ctx.input_params 
#         del output_tensors

#         # Return gradients
#         # Two none for run_function and length (non-tensor arguments)
#         # input_grads for input tensors and parameters

#         testing_output = (None, None) + input_grads
#         print(f"what is the testing shape [class-CheckpointFunction]: {testing_output}")    # (None, None, torch.Size([4, 1024, 320]), torch.Size([4, 77, 768]), torch.Size([4, 1024, 320]), torch.Size([4, 77, 768]))

#         return (None, None) + input_grads
    

class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors

        print(f"what is the Tensor type of input_grads [class-CheckpointFunction]: {input_grads}")
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