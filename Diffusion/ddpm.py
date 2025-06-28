import torch
from torch import nn 
import pytorch_lightning as pl
import numpy as np 
from functools import partial
from contextlib import contextmanager
from einops import rearrange, repeat
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from Diffusion.utils import (
    instantiate_from_config, 
    count_params, 
    exists,
    make_beta_schedule,
    default,
    extract_into_tensor,
    noise_like,
    load_config,
    log_txt_as_img,
    isimage,
    ismap, 
   
)
from vqvae.autoencoder import VQModelInterface

from Ema.ema import LitEma
from Distribution.distribution import DiagonalGaussianDistribution
from tqdm import tqdm

from torchvision.utils import make_grid
from Diffusion.ddim import DDIMSampler

class DiffusionWrapper(pl.LightningModule):

    def __init__(self,
                 diff_model_config,
                 conditioning_key):
        
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditional_key = conditioning_key

        assert self.conditional_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):

        if self.conditional_key is None:
            out = self.diffusion_model(x, t)

        elif self.conditional_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)

        elif self.conditional_key == 'crossattn':
            print(f"what is the data to get [DiffusionWrapper]: >>>> {c_crossattn}")
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)

        elif self.conditional_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)

        elif self.conditional_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)

        else:
            raise NotImplementedError()
        

        return out
    

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


class DDPM(pl.LightningModule):

    # classic DDPM with Gaussian diffusion, in image space 

    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0,
                 v_posterior=0.,    # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta 
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",    # all assuming fixed variance schedules 
                 scheduler_config=None,
                 use_positional_encoding=False,
                 learn_logvar=False,
                 logvar_init=0.
                 ):
        
       
        

        super().__init__()
        

        



        assert parameterization in ["eps", "x0"], 'currently onley supporting "eps" and "x0" '
        self.parameterization = parameterization

        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")

        self.cond_stage_model = None 
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size
        self.channels = channels

        self.use_positional_encodings = use_positional_encoding
        self.model = DiffusionWrapper(diff_model_config=unet_config,
                                      conditioning_key=conditioning_key)
        
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        
        if self.use_ema:
            self.model_ema = LitEma(model=self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}")


        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config


        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)


        self.register_schedule(given_betas=given_betas,
                               beta_schedule=beta_schedule,
                               timestep=timesteps,
                               linear_start=linear_start,
                               linear_end=linear_end,
                               cosine_s=cosine_s)
        
        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,)).to("cuda:0")

        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True).to("cuda:0")




        
    def init_from_ckpt(self):
        pass 



    def register_schedule(self, 
                          given_betas=None,
                          beta_schedule="linear",
                          timesteps=1000,
                          linear_start=1e-4,
                          linear_end=2e-2,
                          cosine_s=8e-3):
        

        if exists(given_betas):
            betas = given_betas

        else:
            betas = make_beta_schedule(schedule=beta_schedule,
                                       n_timestep=timesteps,
                                       linear_start=linear_start,
                                       linear_end=linear_end,
                                       cosine_s=cosine_s)
            

        alphas = 1. - betas 
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps = betas.shape[0]
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end

        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculation for diffusion q(x_t | x_{t-1}) and others 
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculation for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
            1. - alphas_cumprod) + self.v_posterior * betas
        
        self.register_buffer('posterior_variance', to_torch(posterior_variance))

        # below: log calculation clipped because the posterior variance is 0 at the begining of the diffusion chain 
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        ))

        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)
        ))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)
            )

        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))

        else:
            raise NotImplementedError("mu not supported")
        

        # TODO: how to choose this term 

        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)

        assert not torch.isnan(self.lvlb_weights).all()


    
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())

            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")

        try:
            yield None 

        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")



    def forward(self, x, *args, **kwargs):

        # b, c, h, w, device, img_size = *x.shape, x.device, self.image_size 
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()

        return self.p_losses(x, t, *args, **kwargs)
    

    def get_input(self, batch, k):

        print(f"check the Tensor of input Image: >>>>>>>>> {batch}")
        # print(f"check the Tensor of input Image: >>>>>>>>> {k}")

        batch = batch["image"]

        x = batch
        print(f"what is the shape of image in [DDPM-class]: >>>>>> {x.shape}")
        
        # if len(x.shape) == 3:
        #     x = x[..., None]

        # x = rearrange(x, 'b h w c -> b c h w')
        # print(f"what is the shape After the rearrange the shape: {x.shape}")

        
        x = x.to(memory_format=torch.contiguous_format).float()

        return x 
    # ---------------------------------------------------------------------

    def q_sample(self, x_start, t, noise=None):
        # print(f"what is the data to get in [q_sample class]: >>>>> {x_start}")
        # print(f"what is the shape of timestep: {t}")
        x_start = x_start.to("cuda:0")
        # print(f"what is the data to get in [q_sample class]: >>>>> {x_start}")

        noise = default(noise, lambda: torch.randn_like(x_start)).to("cuda:0")
        # print(f"is noise tensor are goes to cuda or not: >>>> {noise}")

        # sqrt_alphas_cumprod_tensor = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        # print(f"what is the shape of [q_sample class]: >>>> {sqrt_alphas_cumprod_tensor}")

        # sqrt_one_minus_alphas_cumprod_tensor = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        # print(f"what is the data to get .... >>>> {sqrt_one_minus_alphas_cumprod_tensor}")

        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape).to("cuda:0") * noise)
    

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    
# ---------------------------------------------------
    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance




    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    


    

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    


    def shared_step(self, batch):
        inputs = self.get_input(batch, self.first_stage_key)
        # print(f"what is the data to get in function [shared_steps]: >>>>> {inputs}")

        x, c = inputs[0], inputs[1]
        # print(f"what is the shape of x in function [shared_steps]: >>>>> {x.shape}")
        # print(f"what is the shape of caption in function [shaared_steps]: >>>>>> {c.shape}")

        loss, loss_dict = self(x, c)
        return loss, loss_dict
    


    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss
    

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)


    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)


    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid
    




    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt







class LatentDiffusion(DDPM):

    """ Main class"""

    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 cond_stage_key="txt",
                 num_timesteps_cond=None,
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 *args,
                 **kwargs
                 ):
        
        
        
        num_timesteps_cond = 1
        self.learning_rate = 1.0e-04
        # first_stage_config = config['model']['params']['first_stage_config']
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std

        # print(f"what is the output of timesteps: {kwargs['timesteps']}")
        assert self.num_timesteps_cond <= 1000

        # for kwargs compatibility after implementation of DiffusionWrapper 
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'

        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None

        ckpt_path = kwargs.pop('ckpt_path', None)
        ignore_keys = kwargs.pop("ignore_keys", [])

        super().__init__(conditioning_key=conditioning_key, 
                         *args,
                         **kwargs,
                        #  unet_config=config['model']['params']['unet_config']
                         )
        
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key

        
        
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1

        except:
            self.num_downs = 0 

        if not scale_by_std:
            self.scale_factor = scale_factor

        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))

        # first_stage_config = config['first_stage_config']
        # cond_stage_config = config['cond_stage_config']
        # print(f"first stage config >>>>>>>>>>>>>>>>>>>> {first_stage_config}")

        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True


    def instantiate_first_stage(self, config):
        # print(f"what i get config: {config}")
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False


    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model


    def register_schedule(self, given_betas=None, beta_schedule="linear", timestep=1000, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timestep, linear_start, linear_end, cosine_s)
    
        self.shorten_cond_schedule = self.num_timesteps_cond > 1 
        if self.shorten_cond_schedule:
            self.make_cond_schedule()


    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps, ), fill_value=self.num_timesteps -1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps -1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids 


    




    @torch.no_grad()
    def get_input(self,
                  batch,
                  k,
                  return_first_stage_outputs=False,
                  force_c_encode=False,
                  cond_key=None,
                  return_original_cond=False,
                  bs=None):
        

        # print(f"what is the data to get batch : >>>>>>>> {batch}")
        # print(f"what is the information in k: >>>> {k}")
        x = super().get_input(batch, k)
        # print(f"what is the input data to get [get_input function]: >>>>>>> {x}")

        if bs is not None:
            x = x[:bs]
        

        x = x.half().cuda()
        # print("what is the shape of input data in [DDPM -class]: >>>>>>>> ", x.shape)

        encoder_posterior = self.encode_first_stage(x)
        # print(f"what is the output to get function [encode_first_stage]: >>>>>> {encoder_posterior}")


        z = self.get_first_stage_encoding(encoder_posterior).detach()
        # print(f"what is the data to get function [get_first_stage_encoding]: >>>>> {z.shape}")

        if self.model.conditional_key is not None:

            if cond_key is None:
                print(f"is this working [cond_key] is None")
                cond_key = self.cond_stage_key
                # print(f"what is the output to return : >>>> {cond_key}")  # 'txt'

            if cond_key != self.first_stage_key:    # 'txt' != 'image'
                if cond_key in ['caption', 'coordination_bbox', 'txt']:
                    xc = batch[cond_key]
                    # print(f"what is the output to get in the caption: >>>> {xc}")

                elif cond_key == 'class_label':
                    xc = batch

                else:
                    xc = super().get_input(batch, cond_key).cuda()

            else:
                xc = x 

            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    c = self.get_learned_conditioning(xc)

                else:
                    c = self.get_learned_conditioning(xc.cuda())

            else:
                c = xc 

            if bs is not None:
                c = c[:bs]

            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shift(batch)
                ckey = __conditioning_keys__[self.model.conditional_key]
                c = {ckey: c,
                     'pos_x': pos_x,
                     'pos_y': pos_y}
                

        else:
            c = None 
            xc = None 

            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shift(batch)
                c = {'pos_x': pos_x,
                     'pos_y': pos_y}
                
        

        z = z.half()
        c = c.half()

        # print(f"what is the dtype of image data: >>>> {z}")
        # print(f"what is the dtype of caption of : >>>> {c}")
                
        out = [z, c]

        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])

        if return_original_cond:
            out.append(xc)

        return out 
            


    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        # print(f"data are comming in the decode function: >>>>> {z.shape}")


        # if prediction codebook indices, convert z to codebook entries 
        if predict_cids:
            if z.dim() == 4:
                # convert one-hot to indices 
                z = torch.argmax(z.exp(), dim=1).long()

            # Get codebook entry 
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)

            # Rearrange tensor dimensions 
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        # scale latent vector by inverse scale factor 
        z = 1. / self.scale_factor * 2 

        # check if using patch-based processing 
        if hasattr(self, 'split_input_params'):

            if self.split_input_params['patch_distributed_vq']:
                # Get kernel size, stride and upscale factor 
                ks = self.split_input_params["ks"]
                stride = self.split_input_params["stride"]
                uf = self.split_input_params["vqf"]

                bs, nc, h, w = z.shape 

                # Adjust kernel size if too large 
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(stride[1], w))


                # Get folding/unfolding operations 
                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                # unfold input into patches 
                z = unfold(z)

                # Reshape to (batch, channels, kernel_h, kernel_w, num_patches)
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))

                # Decode each patch 
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                                                 for i in range(z.shape[-1])
                                                                 ]
                    

                else:
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]
                    

                # stack outputs and apply weighting 
                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reshape back to unfolded form 
                o = o.view((o.shape[0], -1, o.shape[-1]))

                # fold patches back into full image 
                decoded = fold(o)

                # normalize by overlap count 
                decoded = decoded / normalization

            else:

                # non-patch-based decoding 
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                
                else:
                    return self.first_stage_model.decode(z)
                

        else:

            # standard decoding path 
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            
            else:
                return self.first_stage_model.decode(z)
            


# --------------------------------------------
    @torch.no_grad()
    def encode_first_stage(self, x):

        # print(f"check the data have split condition or not: >>>>>> {x}")

        if hasattr(self, "split_input_params"):
            print(f"is this working...")
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]
                stride = self.split_input_params["stride"]
                df = self.split_input_params["vqf"]
                self.split_input_params["original_image_size"] = x.shape[-2:]

                bs, nc, h, w = x.shape

                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print('reducing the kernel....')


                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print('reducing stride ....')

                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)   # (bn, nc * prod(**ks), L)

                # Reshape to img shape 
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1])) # (bn, nc, ks[0], ks[1], L)

                output_list = [self.first_stage_model.encode(z[:, :, :, :, i])
                               for i in range(z.shape[-1])]
                
                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape 
                o = o.view((o.shape[0], -1, o.shape[-1]))   # (bn, nc * ks[0] * ks[1], L)

                # switch crops together 
                decoded = fold(o)
                decoded = decoded / normalization

                return decoded

            else:
                print("is this else condition are working...")
                return self.first_stage_model.encode(x)
            
        else:
            print("what is the condition are working...")
            return self.first_stage_model.encode(x)
        




    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):  # TODO: Load once not every time, shorten code.
        
        """ 
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """

        bs, nc, h, w = x.shape

        # number of crops in image 
        Ly = (h - kernel_size[0]) // stride[0] + 1 
        Lx = (w - kernel_size[1]) // stride[1] + 1 

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size,
                               dilation=1,
                               padding=0,
                               stride=stride
                               )
            
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(h=kernel_size[0],
                                           w= kernel_size[1],
                                           Ly=Ly,
                                           Lx=Lx,
                                           device=x.device).to(x.dtype)
            
            normalization = fold(weighting).view(1, 1, h, w)    # normalize the overflow 
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))


        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                                dilation=1, padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                                dilation=1, padding=0,
                                stride=(stride[0] // df, stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))


        else:
            raise NotImplementedError
        

        return fold, unfold, normalization, weighting



    def get_weighting(self, h, w, Ly, Lx, device):

        weighting = self.delta_border(h, w)
        weighting = torch.clip(weighting, 
                               self.split_input_params['clip_min_weight'],
                               self.split_input_params['clip_max_weight'])
        
        weighting = weighting.view(1, h*w, 1).repeat(1, 1, Ly*Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting,
                                     self.split_input_params["clip_min_tie_weight"],
                                     self.split_input_params["clip_max_tie_weight"])
            
            L_weighting = L_weighting.view(1, 1, Ly*Lx).to(device)
            weighting = weighting * L_weighting

        return weighting
    



    def delta_border(self, h, w):

        """
        :param h: height 
        :param w: width
        :return: normalized distance to image border.
            with min distance = 0 at border and max dist = 0.5 at image center
        """

        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner

        dist_left_up = torch.min(arr, dim=1, keepdim=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdim=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]

        return edge_dist
    




    def meshgrid(self, h, w):

        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr 
    
# ----------------------------------------------------------------------------------------------------------------


    def get_first_stage_encoding(self, encoder_posterior):

        # print(f"Let's know the encoder_posterior: {encoder_posterior}")

        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()

        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior

        elif isinstance(encoder_posterior, tuple):
            z = encoder_posterior[0]



        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented. ")
        
        return self.scale_factor * z 
    


    def forward(self, x, c, *args, **kwargs):

        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device="cuda:0").long()
        print(f"Timestep tensor: >>>>> {t}")

        if self.model.conditional_key is not None:
            assert c is not None
            print(f"what is the data to get the conditioning key: {self.model.conditional_key}")


            if self.cond_stage_trainable:
                print(f"what is the mean of {self.cond_stage_trainable}")
                c = self.get_learned_conditioning(c)

            # if self.shorten_cond_schedule:  # TODO: drop this option 
            #     print(f"what is the meaning of sorten_cond_schedule: >>>> {self.shorten_cond_schedule}")
            #     tc = self.cond_ids[t].cuda()
            #     c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))

        return self.p_losses(x, c, t, *args, **kwargs)



    def get_learned_conditioning(self, c):
        print(f"can get the data: {c}")
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                # print('is this working....')
                c = self.cond_stage_model.encode(c)

                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()

            else:
                # print(f"what is the input i get : >>>>>>> {c.shape}")
                # print(f"what is the input i get : >>>>>>> {c.dtype}")
                c = self.cond_stage_model(c)


        else:
            # print("this else condition are working...")
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)

        return c 
    
# ----------------------------------------------------------------------------

    def p_losses(self, x_start, cond, t, noise=None):

        # Generate noise if not provided 
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        # Create noisy sample at timestep t 
        x_noisey = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Apply denosing model to noisy input
        model_output = self.apply_model(x_noisey, t, cond)
        # print(f"what is the output of model_output function [p_losses]: >>>> {model_output.shape}")   # torch.Size([4, 4, 32, 32])

        # Initialize loss dictionary 
        loss_dict = {}

        # set prefix for logging (train/val)
        prefix = 'train' if self.training else 'val'

        # Determine target based on parameterization 
        if self.parameterization == "x0":
            target = x_start    # Predict original input 

        elif self.parameterization == "eps":
            target = noise  # predict noise 
            # print(f"what is the output to get target variable: >>> {target.shape}") # torch.Size([4, 4, 32, 32])

        else:
            raise NotImplementedError()
        

        # calculate sample loss (per-element)
        loss_simple = self.get_loss(pred=model_output,
                                    target=target,
                                    mean=False).mean([1, 2, 3])
        
        # store mean simple loss 
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        # Get log variance for current timestep 
        print(f"what is the time output function [p_losses]: >>> {t}")
        print(f"which to get logvar: {self.logvar}")
        logvar_t = self.logvar[t]

        # compute gamma-weighted loss (for learnable variance)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t

        # Handle learnable log variance 
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        # Apply weight to simple loss component 
        loss = self.l_simple_weight * loss.mean()

        # calculate variational lower bound loss 
        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))

        # weighting VLB by timestep weights 
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()

        # store VLB loss 
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})

        # Add weighted VLB to total loss 
        loss += (self.original_elbo_weight * loss_vlb)

        # store total loss 
        loss_dict.update({f'{prefix}/loss': loss})


        return loss, loss_dict



    def apply_model(self, x_noisy, t, cond, return_ids=False):

        # Handle conditioning input (convert to dictionary format if needed)
        if isinstance(cond, dict):
            
            # hybrid case: cond is already in expected dict format 
            pass 

        else:

            # wrap single conditioning in list 
            if not isinstance(cond, list):
                cond = [cond]

            # Determine conditioning key based on model type 
            key = 'c_concat' if self.model.conditional_key == 'concat' else 'c_crossattn'
            # format as dictionary with proper key 
            cond = {key: cond}


        # check for patch-based processing (for large images)
        if hasattr(self, 'split_input_params'):
            
            # currently only supports one conditioning type 
            assert len(cond) == 1 

            # Not implemented for return_ids 
            assert not return_ids

            # Get patch parameters 
            ks = self.split_input_params["ks"]  # kernel size (patch_size)
            stride = self.split_input_params["stride"]  # stride for sliding window

            # current spatial dimensions 
            h, w = x_noisy.shape[-2:]

            # parepare folding/unfoling operations 
            fold, unfold, normalization, weighting = self.get_fold_unfold(x_noisy, ks, stride)

            # unfold input into patches 
            z = unfold(x_noisy)     # (batch_size, channels*ks_h*ks_w, num_patches)

            # Reshape to: (batch_size, channels, ks_h, ks_w, num_patches)
            z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))

            # split into list of individual patches 
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]

            # Handle different conditioning types for patches 
            if self.cond_stage_key in ["image", "LR_image", "segmentation", "bbox_img"]:

                # Get conditioning key 
                c_key = next(iter(cond.keys()))
                # Get conditioning values 
                c = next(iter(cond.values()))

                # unfold conditioning similarly 
                c = unfold(c)
                c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))  # [batch_size, channels, ks_h, ks_w, num_patches]

                # Create seperate conditioning dict for each patch
                cond_list = [{c_key: [c[:, :, :, :, i]]} for i in range(c.shape[-1])]


            elif self.cond_stage_key == 'coordinates_bbox':

                assert 'original_image_size' in self.split_input_params, 'BoundingBoxRescaling is missing original_image_size'

                # assuming padding of unfold is always 0 and its dilation is always 1 
                n_patches_per_row = int((w - ks[0]) / stride[0] + 1)
                full_img_h, full_img_w = self.split_input_params['original_image_size']

                # As we are operating on latent, we need the factor from the original image size to the 
                # spatial latent size to properly rescale the crops for regenerating the bbox annotations

                num_downs = self.first_stage_model.encoder.num_resolutions - 1 
                rescale_latent = 2 ** (num_downs)

                # Get top left position of patches as conforming for the bbox tokenizer, therefore we
                # need to rescale the t1 patch cooridinates to be in between (0, 1)

                t1_patch_coordinates = [(rescale_latent * stride[0] * (patch_nr % n_patches_per_row) / full_img_w,
                                         rescale_latent * stride[1] * (patch_nr  // n_patches_per_row) / full_img_h)
                                         for patch_nr in range(z.shape[-1])]
                

                # patch_limits are t1_cooridinates, width and height coordinates as (x_t1, y_t1, h, w)
                patch_limits = [(x_t1, y_t1,
                                 rescale_latent * ks[0] / full_img_w,
                                 rescale_latent * ks[1] / full_img_h) for x_t1, y_t1 in t1_patch_coordinates]
                

                # tokenize crop coordinates for the bounding boxes of the respective patches 
                patch_limits_tknd = [torch.LongTensor(self.bbox_tokenizer._crop_encoder(bbox))[None].cuda()
                                     for bbox in patch_limits]  # list of length 1 with tensors of shape (1, 2)
                
                # print(f"what is the shape of patch_limit_tokenizer number of dim: >>>> {patch_limits_tknd[0].shape}")

                # cut tokenizer crop position from conditionng 
                assert isinstance(cond, dict), "cond must be dict to be fed into model"
                cut_cond = cond["c_crossattn"][0][..., :-2].cuda()
                
                # print(f"what is the shape of cut_cond: >>>>>>>>> {cut_cond.shape}")


                adapted_cond = torch.stack([torch.cat([cut_cond, p], dim=1) for p in patch_limits_tknd])
                adapted_cond = rearrange(adapted_cond, 'l b n -> (l b) n')

                # print(f"what is the shape of Adaptive condition: >>>>>>> {adapted_cond.shape}")

                adapted_cond = self.get_learned_conditioning(adapted_cond)
                # print(f"what is the shape of Adaptive condition [After applying get_learned_condition class]: >>>>>>> {adapted_cond.shape}")

                adapted_cond = rearrange(adapted_cond, '(l b) n d -> l b n d', l=z.shape[-1])
                # print(f"what is the shape of Adaptive condition [After rearring the shape]: >>>>>>> {adapted_cond.shape}")

                cond_list = [{'c_crossattn': [e]} for e in adapted_cond]


            else:

                # Duplicate same conditioning for all patches 
                cond_list = [cond for i in range(z.shape[-1])]  # TODO: make this more efficient 
                      

            # Apply model to each patch independently 
            output_list = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]

            # stack outputs while preserving gradients 
            o = torch.stack(output_list, axis=-1)
            # Apply weighting for overlapping regions 
            o = o * weighting
            # Reshape back to unfolded format 
            o = o.view((o.shape[0], -1, o.shape[-1]))

            # Fold patches back into complete feature map 
            x_recon = fold(o) / normalization   # normalize by overlap count 


        else:
            
            # Standard full-image processing 
            x_recon = self.model(x_noisy, t, **cond)

        # Handle different return type 
        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]   # Return only main output
        
        else:
            return x_recon  # Return full outputs


# ----------------------------------------------------

# ----------------------------------------------------------------------------

    def _predict_eps_from_xstart(self,
                                 x_t,
                                 t,
                                 pred_xstart):
        
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    


    def _rescale_annotations(self,
                             bboxes,
                             crop_coordinates):
        

        def rescale_bbox(bbox):

            x0 = torch.clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2])
            y0 = torch.clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3])

            w = min(bbox[2] / crop_coordinates[2], 1 - x0)
            h = min(bbox[3] / crop_coordinates[3], 1 - y0)

            return x0, y0, w, h 

        return [rescale_bbox(b) for b in bboxes]
    


    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt
    

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")
    







    
    


if __name__ == "__main__":

    from torch.utils.data import Dataset, DataLoader

    config = load_config(config_path="Diffusion/config.yaml")
    # print(config['model']['params']['first_stage_config'])

    

        
    model_params = config['model']['params']



    model = LatentDiffusion(**model_params).half().cuda()
    

    
    from Diffusion.data.dataset import DataModuleFromConfig
    
    
    

    data_moduler = DataModuleFromConfig(batch_size=32,
                                   train=config['data']['params']['train'],
                                   num_workers=4,
                                   use_worker_init_fn=True,
                                   )

    data_loader = data_moduler.train_dataloader()
  
    for batch in data_loader:
    #     batch = batch['image']
    #     batch = batch[1]
    #     print(f"check the shape of image: {batch.shape}")

        

        model.get_input(batch=batch,
                        k='image')
        


    from vqvae.autoencoder import VQModel

    class YourDiffusionModel(nn.Module):

        def __init__(self, 
                     vq_vae_ckpt_path):
            
            super().__init__()

            # Load vq-vae from checkpoint 
            self.first_stage_model = VQModel.load_from_checkpoint(vq_vae_ckpt_path)
            self.first_stage_model.eval()


            self.split_input_params = {
                "patch_distributed_vq": True,
                "ks": (128, 128),
                "stride": (64, 64),
                "vqf": 4, 
                "clip_min_weight": 0.01,
                "clip_max_weight": 1.0
            }


    




        
        

    





            

        
