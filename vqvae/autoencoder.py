import pytorch_lightning as pl 
from Encoder_Decoder.encoder import Encoder, Decoder
from .quantize import VectorQuantize2 as VectorQuantizer
import torch 
from Ema.ema import LitEma
import torch.nn as nn 
from vqvae.utils import instantiate_from_config
import torch.optim as optim 
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer
# from .dataset import VQVAEDataset
from contextlib import contextmanager
import numpy as np 






class VQModel(pl.LightningModule):

    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 lossconfig,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key='image',
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False,
                 use_ema=False):
        
        """ 
        Vector Quantized Variational Autoencoder (VQ-VAE) model.

        Args:
            ddconfig (dict): Configuration dictionary for encoder/decoder 
            n_embed (int): Number of embeddings in the codebook 
            embed_dim (int): Dimension of each codebook embedding 
            lossconfig (bool/dict): Loss function configuration 
            ckpt_path (str): Path to checkpoint for loading pretrained weights 
            ignore_keys (list): Keys to ignore when loading from checkpoint 
            image_key (str): Keys for accessing image in the batch 
            colorize_nlabels (int): Number of labels for colorization (optional)
            monitor (str): Metric to monitor for checkpointing 
            batch_resize_range (tuple): Range for per-batch resizing (optional)
            scheduler_config (dict): Learning rate scheduler configuration
            lr_g_factor (float): Learning rate multiplier for generator 
            remap (str): path to remapping file for codebook indices 
            sane_index_shape (bool): Whether to reshape indices sensibly 
            use_ema (bool): Whether to use exponential moving average
        """


        

        super().__init__()
        # Basic configuration
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key

        self.automatic_optimization = False # automatic optimization 

        # Core VQ-VAE components
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(n_e=n_embed,
                                        e_dim=embed_dim,
                                        beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        
        # Conv layers for pre/post quantization
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)

        # loss 
        self.loss = instantiate_from_config(config=lossconfig)
       

        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        # Optional colorization Setup
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int 
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

        # Training monitoring 
        if monitor is not None:
            self.monitor = monitor


        # Dynamic batch resizing 
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")


        # Exponential Moving Average setup
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}")

        # Checkpoint loading 
        if ckpt_path is not None:
            pass 


        # Learning rate configuration 
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor


    def encode(self, x):
        h = self.encoder(x)
        print("Encoder shape : ", h.shape)   

        h = self.quant_conv(h)
        print("H: ", h.shape)
        quant, emb_loss, info = self.quantize(h)

        print("quant: ", quant.shape)
        print("emb_loss: ", emb_loss.shape)
        print("info: ", info.shape)
        return quant, emb_loss, info 
    

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec 
    

    
    def forward(self, input, return_pred_indices=False):
        print("shape of input tensor: ", input.shape)
        quant, diff, (_, _, ind) = self.encode(input)


        dec = self.decode(quant)

        if return_pred_indices:
            return dec, diff, ind 
        
        return dec, diff 
    

    def training_step(self, batch, batch_idx):


        # get optimizers 
        opt_ae, opt_disc = self.optimizers()

        # x = batch['image']
        x = self.get_input(batch, self.image_key)
        
        # Foward pass - returns continous latents 
        h = self.encode(x)

        # Decode with quantization (normal training mode)
        reconstructions = self.decode(h)


        # Get quantization (normal training mode)
        if isinstance(h, tuple):
            h = h[0]
        _, codebook_loss, _ = self.quantize(h)

        # calculate loss 
        aeloss, log_dict_ae = self.loss(
            codebook_loss = codebook_loss,
            inputs=x,
            reconstructions=reconstructions,
            optimizer_idx=1,
            global_step=10,
            split="train",
            last_layer = self.get_last_layer()
        )

        # Generator update 
        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()

        # Discriminator update (if using adversarial loss)
        discloss, log_dict_disc = self.loss(
            codebook_loss=codebook_loss,
            inputs=x,
            reconstructions=reconstructions,
            optimizer_idx=1,
            global_step=1,
            split="train",
            last_layer = self.get_last_layer()
        )

        opt_disc.zero_grad()
        self.manual_backward(discloss)
        opt_disc.step()

        # Logging 
        self.log_dict(log_dict_ae, prog_bar=False)
        self.log_dict(log_dict_disc, prog_bar=False)

        return aeloss + discloss


        

    

    

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict


    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        print("data:", x)
        xrec, qloss, ind = self(x, return_pred_indices=True)

        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer = self.get_last_layer(),
                                        split="val"+suffix,
                                        )
        
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            )
        
        self.log(f"val{suffix}/rec_loss", log_dict_ae[f"val{suffix}/rec_loss"],
             prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        # Compute and log total loss 
        total_loss = aeloss + discloss
        self.log(f"val{suffix}/total_loss", total_loss, 
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        return log_dict_ae

    

    

    def configure_optimizers(self):
        # Create optimizers
        lr = self.hparams.get('lr', 4.5e-6)
        opt_ae = optim.Adam(
            list(self.encoder.parameters()) + 
            list(self.decoder.parameters()) + 
            list(self.quant_conv.parameters()) + 
            list(self.post_quant_conv.parameters()),
            lr=lr, betas=(0.5, 0.9)
        )
        
        opt_disc = optim.Adam(
            self.loss.discriminator.parameters(),
            lr=lr, betas=(0.5, 0.9)
        )
        
        # Create scheduler
        scheduler_ae = {
            'scheduler': ReduceLROnPlateau(opt_ae, mode='min', factor=0.5, patience=5),
            'monitor': 'val/rec_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        
        return [opt_ae, opt_disc], [scheduler_ae]
    

    def get_last_layer(self):
        return self.decoder.conv_out.weight
    
   

    def log_images(self, batch, plot_ema=False, **kwargs):

        assert isinstance(batch, (dict, torch.Tensor)), "Batch must be dict or tensor"

        # Initialize logging dictionary
        log = dict()

        # Debug print to check batch structure (typically removed in production)
        print(f"Shape of batch['image']: {batch}")

        x = batch[self.image_key] if isinstance(batch, dict) else batch 
        x = x.to(self.device)

        
            
        
        # Get model reconstructions 
        xrec, _ = self(x)   # Forward pass through model

        # Handle multi-channel images (more than 3 channels)
        if x.shape[1] > 3:  # check the channel dim

            # Safety check that reconstruction matches input channels
            assert xrec.shape[1] > 3, "Image channels are more than 3"
            # Convert to RGB for visualization
            x = self.to_rgb(xrec)
            xrec = self.to_rgb(xrec)

        # Store results in log dictionary 
        log['inputs'] = x 
        log['reconstructions'] = xrec

        # Optional EMA version processing 
        if plot_ema:
            with self.ema_scope():  # Context manager for EMA weights
                xrec_ema, _ = self(x)   # Forward pass with EMA weights
                # convert EMA reconstruction to RGB
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions"] = xrec_ema   # Store EMA version

        return log 
    



    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))

        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min()) / (x.max()-x.min()) -1.
        return x 
    

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")

        try:
            yield None 

        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")


    def get_input(self, batch, k):

        """ 
        Processes and potentially resizes input data from a batch dictionary.

        This method:
        1. Extract input data from the batch 
        2. Ensure proper tensor dimensions and memory format 
        3. Optionally applies dynamic resizing during initial training steps 
        4. Returns a processed, detached tensor 

        Args:
            batch (dict): Dictionary containing the input batch data 
            k (str): Key to access the input data in the batch dictionary 

        Returns:
            torch.Tensor: Processed input tensor with shape (B, C, H, W)
        """

        # Get the input data from batch dictionary
        x = batch[k]
        print("shape of X in get Input: --->", x.shape)

        # Add channel dimension if input is 3D (assumed to be [B, H, W])
        # Ensure input has the correct shape [B, C, H, W]
        if x.ndim == 4 and x.shape[1] != 3:  # If channels are not in the second dimension
            x = x.permute(0, 3, 1, 2)  # Permute to [B, C, H, W]
        

        # Reorder dimension to standard [B, C, H, W] format and ensure contiguous memory
        x = x.to(memory_format=torch.contiguous_format).float()

        # Dynamic resizing logic (if batch_resize_range is specified)
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0] # Minimum resize dim
            upper_size = self.batch_resize_range[1] # Max resize dim 

            # Special handling for first few training steps 
            if self.global_step <= 4:
                # Use maximum size initially to avoid OOM error from upscaling 
                new_resize = upper_size

            else:
                # Readomly select a size from the allowed range (in steps of 16)
                new_resize = np.random.choice(np.arange(lower_size, upper_size+16, 16))

            # Apply resizing if needed 
            if new_resize != x.shape[2]:    # check if height needs resizing 
                x = nn.functional.interpolate(x, size=new_resize, mode='bicubic')

            # Detach from computation graph to prevent backprop through resizing
            x = x.detach()

        return x 
    



        
    


    




class VQModelInterface(VQModel):

    def __init__(self, 
                 embed_dim,
                 *args,
                 **kwargs):
        super().__init__(embed_dim=embed_dim,
                         *args,
                         **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        print(f"Input shape to encode: {x.shape}")
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h, torch.tensor(0.0).to(h.device), (None, None, None)
    

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer 
        if not force_not_quantize:
            if isinstance(h, tuple):
                h = h[0]

            print("h: -->",  h)
                # Quantize and get all outputs
            quant, emb_loss, info = self.quantize(h)

        else:
            quant = h 

        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec 
    





    
        

        