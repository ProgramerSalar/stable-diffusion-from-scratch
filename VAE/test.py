import torch 
from torch import nn 
from VAE.autoencoder import AutoEncoderKL
from Dataset.lsun import LSUNBase
from torch.utils.data import DataLoader
import yaml






# -----------------------------------------------------------------------------------------------------------



# path = "diffusion_pytorch_model.fp16.safetensors"


# from safetensors.torch import load_file

# loaded = load_file(path)
# # print(loaded)

# for key, tensor in loaded.items():
#     print("key: ", key)
#     print("tensor: ", tensor.shape)

#     exit()

# -----------------------------------------------------


path = "lightning_logs/version_0/checkpoints/epoch=0-step=961.ckpt"





model = torch.load(path, map_location="cuda")["state_dict"]
for k, v in model.items():

    print("what is meaning of key:", k)
    print("what is the value: ", v.shape)
    print("Tensor :", v)

    exit()
    

