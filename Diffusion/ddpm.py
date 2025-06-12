import torch
from torch import nn 
import pytorch_lightning as pl
import numpy as np 
from functools import partial
from contextlib import contextmanager
from einops import rearrange

from Diffusion.utils import (
    instantiate_from_config, 
    count_params, 
    exists,
    make_beta_schedule,
    default,
    extract_into_tensor,
    noise_like,
    load_config
)
from vqvae.autoencoder import VQModelInterface

from Ema.ema import LitEma
from Distribution.distribution import DiagonalGaussianDistribution
from tqdm import tqdm

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


        
    def init_from_ckpt(self):
        pass 



    def register_schedule(self, 
                          given_betas=None,
                          beta_schedule="linear",
                          timestep=1000,
                          linear_start=1e-4,
                          linear_end=2e-2,
                          cosine_s=8e-3):
        

        if exists(given_betas):
            betas = given_betas

        else:
            betas = make_beta_schedule(schedule=beta_schedule,
                                       n_timestep=timestep,
                                       linear_start=linear_start,
                                       linear_end=linear_end,
                                       cosine_s=cosine_s)
            

        alphas = 1. - betas 
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps = betas.shape 
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
            lvlb_weight = self.betas ** 2 / (
                2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)
            )

        elif self.parameterization == "x0":
            lvlb_weight = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))

        else:
            raise NotImplementedError("mu not supported")
        

        # TODO: how to choose this term 

        lvlb_weight[0] = lvlb_weight[1]
        self.register_buffer('lvlb_weights', lvlb_weight, persistent=False)

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

        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]

        x = x.to(memoery_format=torch.contiguous_format).float()

        return x 
    # ---------------------------------------------------------------------

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
    

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

    def p_losses(self, x_start, t,  noise=None):

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise 

        elif self.parameterization == "x0":
            target = x_start

        else:
            raise NotImplementedError(f"Parameterization {self.parameterization} not yet supported.")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = "train" if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weigh[t] * loss).mean()    
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss.dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict
    
# ---------------------------------------------------

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img
    
    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    

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








class LatentDiffusion(DDPM):

    """ Main class"""

    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
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
        
        
        
        num_timesteps_cond = config['model']['params']['num_timesteps_cond']
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
                         unet_config=config['model']['params']['unet_config'])
        
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable

        
        
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1

        except:
            self.num_downs = 0 

        if not scale_by_std:
            self.scale_factor = scale_factor

        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))

        first_stage_config = config['model']['params']['first_stage_config']
        cond_stage_config = config['model']['params']['cond_stage_config']

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














    def forward(self, x, c, *args, **kwargs):

        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()

        if self.model.conditional_key is not None:
            assert c is not None

            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)

        return self.p_loss()



    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)

                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()

            else:
                c = self.cond_stage_model(c)


        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)

        return c 
    


    
    


if __name__ == "__main__":

    from torch.utils.data import Dataset, DataLoader

    config = load_config(config_path="Diffusion/config.yaml")
    # print(config['model']['params']['first_stage_config'])

    # 2. create Dataset 
    class TextImageDataset(Dataset):

        def __init__(self, num_samples=1000):
            self.data = []

            for i in range(num_samples):
                self.data.append({
                    'image': torch.randn(3, 256, 256),
                    'caption': f"A beautiful landscape with mountains {i}"
                })



        def __len__(self):
            return len(self.data)
        

        
        def __getitem__(self, idx):
            return self.data[idx]
        

    # 3. Data collator 
    def collate_fn(batch):
        return {
            "image": torch.stack([item["image"] for item in batch]),
            "caption": [item["caption"] for item in batch]
        }
    

    # 4. Data Module 
    class DiffusionDataModule(pl.LightningDataModule):
        def __init__(self, batch_size=4):
            super().__init__()
            self.batch_size = batch_size
            
        def train_dataloader(self):
            dataset = TextImageDataset(num_samples=1000)
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_fn
            )
        
        def val_dataloader(self):
            dataset = TextImageDataset(num_samples=100)
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                collate_fn=collate_fn
            )

        




    model = LatentDiffusion(
                            # first_stage_config=config['model']['params']['first_stage_config'],
                            #  cond_stage_config=config['model']['params']['cond_stage_config'],
                            #  num_timesteps_cond=config['model']['params']['num_timesteps_cond'],
                             *config
                             )
    
    
    

    





            

        
