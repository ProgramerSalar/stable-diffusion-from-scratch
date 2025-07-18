import torch 
from torch import nn 
from Encoder_Decoder.encoder import Encoder, Decoder
import importlib
import yaml
from Distribution.distribution import DiagonalGaussianDistribution
from Dataset.lsun import LSUNBase
import os 
import pytorch_lightning as pl 
from torch.utils.data import DataLoader




def instantiate_from_config(config):
    print("config: ", config)

    return get_obj_from_str(string=config["target"])(**config.get("params", dict()))
    

def get_obj_from_str(string, reload=False):

    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)

    return getattr(importlib.import_module(module, package=None), cls)
    


class AutoEncoderKL(pl.LightningModule):

    

    def __init__(self,
                 ddconfig,
                 embed_dim,
                 lossconfig=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 colorize_nlabels=None,
                 monitor=None):
        
        super().__init__()
        
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        # add the learning rate
        self.learning_rate = 4.5e-06
        

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


        # Add default loss 
        if lossconfig is None:
            self.loss = self.default_loss

        else:
            self.loss = instantiate_from_config(lossconfig)


    def init_from_checkpoint(self, 
                             path,
                             ignore_keys=list()):
        
        sd = torch.load(path, map_location="cuda")["state_dict"]
        print("sd: ", sd)

        for name, tensor in sd.items():
            # print("tensor: ", tensor)
            if tensor == "nan":
                print("Tensor is NaN")

        # verify NaN exists in state_dict 
        for name, tensor in sd.items():
            print(f"NaN found in: {name}, shape: {tensor.shape}")

        # check for Inf values too 
        for name, tensor in sd.items():
            if torch.isinf(tensor).any():
                print(f"Inf found in: {name}, shape: {tensor.shape}")
        





    def encode(self, x):
        # [(1, 3, 256, 256)] -> ([1, 6, 254, 254])
        print(f"what is the data to get [class-AutoEncoderKL]: {x}")
        h = self.encoder(x)
        # ([1, 6, 254, 254]) -> ([1, 6, 254, 254])
        moments = self.quant_conv(h)
        # ([1, 6, 254, 254]) -> <Distribution.distribution.DiagonalGaussianDistribution object at 0x000001C7343E6AB0>
        posterior = DiagonalGaussianDistribution(parameters=moments)  

        return posterior
    

    def decode(self, z):
        # torch.Size([1, 3, 254, 254]) -> torch.Size([1, 3, 254, 254])
        # print("what is the shape of z: ", z)
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
            z = z
            
           

        else:
            z = posterior.mode()
            z = z

        
        dec = self.decode(z)
        # print("what is shape of decoder: ", dec.shape)
        

        
        return dec, posterior
    

    
    

# ----------------------------------------------------------------------------------------------------------------
    #  If you have Discriminator then apply this part 

    def get_input(self, batch):

        return batch 
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight
    
    
    def training_step(self, batch, batch_idx,):
      
        inputs = self.get_input(batch)
        reconstructions, posterior = self(inputs)


        

        # train with discriminator
        discloss, log_dict_disc = self.loss(inputs,
                                            reconstructions=reconstructions,
                                            posteriors=posterior,
                                            optimizer_idx=1,
                                            global_step=self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="train"
                                                )
            
        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        return discloss
        


    

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch)
        reconstructions, posterior = self(inputs)

        
        
        discloss, log_dict_disc = self.loss(inputs,
                                            reconstructions=reconstructions,
                                            posteriors=posterior,
                                            optimizer_idx=1,
                                            global_step=self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val")
        
        
        self.log("val/disc_loss", log_dict_disc["val/disc_loss"])
        
        
        self.log_dict(log_dict_disc)
        
        return self.log_dict
    

    def configure_optimizers(self):
        lr = self.learning_rate
        
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr,
                                    betas=(0.5, 0.9))
        

        

        return [opt_disc], []
    
    
# ------------------------------------------------------------------------------------------------------------------------

    # ## IF you don't have the discriminator then you use this code 

    # def get_input(self, batch):

    #     return batch 
    
    # def get_last_layer(self):
    #     return self.decoder.conv_out.weight
    

    # def default_loss(self, 
    #                  inputs,
    #                  reconstructions, 
    #                  posterior, 
    #                  global_step,
    #                  last_layer,
    #                  split
    #                  ):
        
    #     # Reconstruction loss (MSE)
    #     rec_loss = nn.functional.mse_loss(inputs, reconstructions)

    #     # KL Divergance loss 
    #     kl_loss = posterior.kl().mean()
    #     total_loss = rec_loss + 0.5 * kl_loss

    #     # Logging 
    #     log_dict = {
    #         f"{split}/rec_loss": rec_loss.detach(),
    #         f"{split}/kl_loss": kl_loss.detach(),
    #         f"{split}/total_loss": total_loss.detach()
    #     }

    #     return total_loss, log_dict
    

    # def training_step(self, batch, batch_idx,):
    #     inputs = self.get_input(batch)
    #     reconstructions, posterior = self(inputs)


        
    #         # train encoder+decoder+logvar 
    #     aeloss, log_dict_ae = self.loss(inputs,
    #                                         reconstructions=reconstructions,
    #                                         posterior=posterior,
    #                                         global_step=self.global_step,
    #                                         last_layer=self.get_last_layer(),
    #                                         split="train")
            
    #     self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
    #     self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)

    #     return aeloss
        
        

        
        


    

    # def validation_step(self, batch, batch_idx):
    #     inputs = self.get_input(batch)
    #     reconstructions, posterior = self(inputs)

    #     aeloss, log_dict_ae = self.loss(inputs,
    #                                     reconstructions=reconstructions,
    #                                     posterior=posterior,
    #                                     global_step=self.global_step,
    #                                     last_layer=self.get_last_layer(),
    #                                     split="val")
        
        
        
    #     self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
    #     self.log_dict(log_dict_ae)
    
        
    #     return self.log_dict
    

    # def configure_optimizers(self):
    #     lr = self.learning_rate
    #     opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
    #                               list(self.decoder.parameters()) + 
    #                               list(self.quant_conv.parameters()) + 
    #                               list(self.post_quant_conv.parameters()),
    #                               lr=lr,
    #                               betas=(0.5, 0.9)
    #                               )
        
        
        

        

    #     return [opt_ae], []

    


    




class IdentityFirstStage(torch.nn.Module):

    def __init__(self,
                 *args,
                 vq_interface=False,
                 **kwargs):
        
        self.vq_interface = vq_interface
        super().__init__()


    def encode(self, x, *args, **kwargs):
        return x 
    

    def decode(self, x, *args, **kwargs):
        return x 
    

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        
        return x 
    

    def forward(self, x, *args, **kwargs):
        return x 


 
        

        

    





if __name__ == "__main__":

    # # # Enable expandable segments to reduce fragmentation
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # torch.cuda.empty_cache()

    # # Use FP16 if possible
    # torch.backends.cudnn.benchmark = True

    # x = torch.randn(1, 3, 256, 256).half().cuda()


    # config = "config/config.yaml"

    # # Load the YAML file 
    # with open(config, 'r') as file:
    #     config = yaml.safe_load(file)

    
    # autoencoder = AutoEncoderKL(ddconfig=config['model']['params']['ddconfig'],
    #                             embed_dim=config['model']['embed_dim']).to("cuda").half().cuda()
    
    # # print(autoencoder)
    # recon_x, posterior = autoencoder(x)
    # print("reconstructed shape of data: ", recon_x.shape)
    # print("posterior: ", posterior)

    
    # # Run with mixed precision 
    # with torch.amp.autocast('cuda'):
    #     output = autoencoder(x)
    #     output = output[0]
        
    # print("Tensor of output: ", output.shape)

# ------------------------------------------------------------------------------------------------------------------------------------

    # x = torch.randn(4, 3, 256, 256).half()
    # y = torch.randn(4, 3, 256, 256).half()

    data_root_train = "Dataset/Data/train"
    data_root_val = "Dataset/Data/val"

    x = LSUNBase(data_root_train)
    y = LSUNBase(data_root_val)


    


    train_data = DataLoader(x, batch_size=4, pin_memory=True, pin_memory_device="cuda")
    test_data = DataLoader(y, batch_size=4 ,pin_memory=True, pin_memory_device="cuda")
    # print(next(iter(train_data)))

    config = "config/vae_config/kl-f4.yaml" 

    # Load the YAML file 
    with open(config, 'r') as file:
        config = yaml.safe_load(file)

    
    autoencoder = AutoEncoderKL(ddconfig=config['model']['params']['ddconfig'],
                                embed_dim=config['model']['embed_dim'],
                                monitor="val/total_loss",
                                lossconfig=config['model']['params']['lossconfig']
                                # lossconfig=None,
                                )
    
    # autoencoder = autoencoder(x)
    # print("AutoEncoder: ", autoencoder)

    

    trainer = pl.Trainer(
        max_epochs=10,
        devices=1, 
        accelerator="gpu",
        precision="16-mixed"    # Enable automatic mixed precision
        # enable_progress_bar=True,
        # progress_bar_refresh_rate=1,
        # checkpoint_callback=True
    )



    trainer.fit(
        autoencoder,
        train_dataloaders=train_data,
        val_dataloaders=test_data
    )



    


        


