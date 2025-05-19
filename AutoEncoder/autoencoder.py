import torch 
from torch import nn 
from Encoder_Decoder.encoder import Encoder, Decoder
import importlib
import yaml
from Distribution.distribution import DiagonalGaussianDistribution


def instantiate_from_config(config):

    return get_obj_from_str(string=config["target"])(**config.get("params", dict()))
    

def get_obj_from_str(string, reload=False):

    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)

    return getattr(importlib.import_module(module, package=None), cls)
    


class AutoEncoderKL(nn.Module):

    pass

    def __init__(self,
                 ddconfig,
                 embed_dim,
                 lossconfig=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None):
        
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        

        assert ddconfig["double_z"], "make sure `double_z: True` "
        self.quant_conv = nn.Conv2d(in_channels=2*ddconfig["z_channels"], 
                                    out_channels=2 * embed_dim, 
                                    kernel_size=1)
        
        # print(self.quant_conv)

        self.post_quant_conv = nn.Conv2d(in_channels=embed_dim,
                                         out_channels=ddconfig["z_channels"],
                                         kernel_size=1)
        

        self.embed_dim = embed_dim

        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int 
            self.register_buffer(name="colorize", 
                                 tensor=torch.randn(3, colorize_nlabels, 1, 1))
            
        if monitor is not None:
            self.monitor = monitor 

        if ckpt_path is not None:
            pass 


    def encode(self, x):
        # [(1, 3, 256, 256)] -> ([1, 6, 254, 254])
        h = self.encoder(x)
        # ([1, 6, 254, 254]) -> ([1, 6, 254, 254])
        moments = self.quant_conv(h)
        # ([1, 6, 254, 254]) -> <Distribution.distribution.DiagonalGaussianDistribution object at 0x000001C7343E6AB0>
        posterior = DiagonalGaussianDistribution(parameters=moments)  

        return posterior
    

    def decode(self, z):
        # torch.Size([1, 3, 254, 254]) -> torch.Size([1, 3, 254, 254])
        z = self.post_quant_conv(z)     
        # torch.Size([1, 3, 254, 254]) -> torch.Size([1, 3, 254, 254])
        dec = self.decoder(z)
        return dec 

        
            

    
    def forward(self, input, sample_posterior=True):
        # [(1, 3, 256, 256)] -> <Distribution.distribution.DiagonalGaussianDistribution object at 0x000001C7343E6AB0>
        posterior = self.encode(input)

        # <Distribution.distribution.DiagonalGaussianDistribution object at 0x000001C7343E6AB0> -> torch.Size([1, 3, 254, 254])
        if sample_posterior:
            z = posterior.sample()
            print("what is the shape of z [If sample_posterior is enabled]: ", z.shape)

        else:
            z = posterior.mode()

        # torch.Size([1, 3, 254, 254]) -> 
        dec = self.decode(z)


        return dec, posterior
    





if __name__ == "__main__":

    x = torch.randn(1, 3, 256, 256)

    config = "config/config.yaml"

    # Load the YAML file 
    with open(config, 'r') as file:
        config = yaml.safe_load(file)

    
    autoencoder = AutoEncoderKL(ddconfig=config['model']['params']['ddconfig'],
                                embed_dim=config['model']['embed_dim'])
    # print(autoencoder)
    cuda = torch.device("cuda:0")
    output = autoencoder.forward(x)

    print(output)
    


        


