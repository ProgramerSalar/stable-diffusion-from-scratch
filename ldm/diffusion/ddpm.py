import torch 
import torch.nn as nn 
import pytorch_lightning as pl
import numpy as np 
from functools import partial
from contextlib import contextmanager
from tqdm import tqdm
from einops import rearrange, repeat
from torchvision.utils import make_grid
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.optim.lr_scheduler import LambdaLR


from ..utils import (
    instantiate_from_config,
    count_params,
    exists,
    default,
    mean_flat,
    log_txt_as_img,
    isimage,
    ismap
    )

from modules.ema import LitEma
from modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from modules.distributions.distributions import DiagonalGaussianDistribution, normal_kl
from models.autoencoder import VQModelInterface, AutoencoderKL, IdentityFirstStage
from diffusion.ddim import DDIMSampler




class DDPM(pl.LightningModule):

    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",  # 
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
                given_bitas=None,
                original_elbo_weight=0.,
                v_posterior=0.,
                l_simple_weight=1.,
                conditioning_key=None,
                parameterization="eps",
                scheduler_config = None,
                use_postional_encoding=False,
                learn_logvar=False,
                logvar_init=0.
                 ):
        

        """ 
            beta_schedule = "linear" : when `beta_schedule="linear"` it means that the betas (noise variance) will increase linearly
                from linear_start to linear_end over the timesteps. This is a common and straightforward way to define the noise
                schedule in DDPM.

            monitor="val/loss" : that val loss `val/loss` is the primarly metric that the training process will observe to assess model 
                performance and guide decisions like saving checkpointing or early stoping.

            
            log_every_t=100 : that the model will record and potentially visualize the state of the image generation process every 100 times steps,
                allowing for monitoring and analysis of the diffusion process.

            clip_denoised=True : The purpose of clipping is to ensure that the reconstructed image values remain within a valid range,
                preventing values from exploding or becoming unrealistic. during the iterative denoising process.
                This can help stablize training and improve the quality of the generated samples.

            cosine_s=8e-3 : when `beta_schedule` is set to `"cosine"` the `make_beta_schedule` function uses `cosine_s` 
                to create a non-linear variance schedule that follows a cosine curve. This specific type of 
                schedule is known to sometimes perform better than a linear schedule. especially for very long 
                diffusion chains, as it allows for a more gradual increase in noise in the early steps and a faster 
                increase in later steps. The `8e-3` value (which is 0.008) is a commonly used small offset to prevent division by zero and shape the curve.


            given_bitas=None : the beta schedule will be automatically generated based on the chosen `beta_schedule` type if you were to provide a tensor of beta
                values here, the model would use those custom values instead.

            original_elbo_weight=0. : the `loss_vlb` component of the loss function is not contributing to the total loss 
                during training. In many DDPM implementations, especially for tasks like image generation, the "simple" loss (eg. L1 or L2 loss between 
                the predicted noise and the actual noise) is found to be sufficient or even preferred for good sample quality. The VLB term can 
                sometimes be diffucult to optimize or may not provide significant benefits in practice. hence it's often weighted to zero by default.

                
            v_posterior=0. : the `v_posterior` is the weight for choosing the posterior variance, where `sigma = (1 - v) * beta_tilde + v * beta` 
                when `v_posterior` is `0`. the posterior variance defaults to `betas * (1. - alphas_cumprod_prev) / (1. - alpha_cumprod) 
                if `v_posterior` were `1.` 
                the posterior variance would simply by `betas` 
                the parameters allows for flexiblity in how the variance is modeled. potentially impacting the stability and quality of the generated samples.


            l_simple_weight=1. : the "simple` loss contributes directly and fully to the total loss without any scaling. in many diffusion models
                this simple noise prediction loss is the primary loss component use for training and often, the model 
                fucuses solely on minimizing the term for effecitive generation.

            parameterization="eps" : the model learns to remove the noise directly when then implicitly allows to reconstruct the clean image. 
                the choice can influence the stability and training and the quality of the generated samples.

        """

        super().__init__()

        assert parameterization in ["eps", "x0"], 'currently only suporting "eps" and "x0" '
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")

        self.cond_stage_model = None 
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size
        self.channels = channels
        self.use_postional_encodings = use_postional_encoding
        
        # <------------- DiffusionWrapper ---------------->
        self.model = DiffusionWrapper(diff_model_config=unet_config,
                                      conditioning_key=conditioning_key)
        count_params(self.model, verbose=True)


        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}")


        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.schedule_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path,
                                ignore_keys=ignore_keys,
                                only_model=load_only_unet)
            


        self.register_schedule(given_betas=given_bitas,
                               beta_schedule=beta_schedule,
                               timesteps=timesteps,
                               linear_start=linear_start,
                               linear_end=linear_end,
                               cosine_s=cosine_s)
        
        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(size=(self.num_timesteps,),
                                 fill_value=logvar_init).to("cuda:0")
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar,
                                       requires_grad=True).to("cuda:0") 

           
        
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

        timesteps = betas.shape 
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculates for diffusion q(x_t | x_{t-1}) and others 
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculation for posterior q(z_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
            1. - alphas_cumprod) + self.v_posterior * betas
        self.register_buffer('posterior_veriance', to_torch(posterior_variance))

        # log calculation clipped because the posterior variance is 0 at the begining of the diffusion chain 
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                2 * self.posterior_veriance * to_torch(alphas) * (1 - self.alphas_cumprod)
            )

        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod) / (2. * 1 - torch.Tensor(alphas_cumprod)))

        else:
            raise NotImplementedError("we are supporting on eps or x0 parameterization")
        

        # TODO: How to choose this term 
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all(), 'make sure lvlb_weights are found the nan values.'


    
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
                    print(f"{context}: Restored training weights.")



    def init_from_ckpt(self, 
                       path,
                       ignore_keys=list(),
                       only_model=False):
        
        sd = torch.load(path, map_location="cpu")

        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startwith(ik):
                    print("Deleting key {k} form state_dict.")
                    del sd[k]

        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")

        if len(missing) > 0:
            print(f"Missing keys: {missing}")

        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected}")



    def q_mean_variance(self, x_start, t):

        """ 
            Get the distribution q(x_t | x_0)
            :param x_start: the [N x C x ...] tensor of noiseless inputs.
            :param t: the number of diffusion steps (minus 1). Here, 0 means one step 
            :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """

        mean = (extract_into_tensor(a=self.sqrt_alphas_cumprod, t=t, x_shape=x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)

        return mean, variance, log_variance
    


    def predict_start_from_noise(self, 
                                 x_t,
                                 t,
                                 noise):
        
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - 
            extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    

    def q_posterior(self,
                    x_start,
                    x_t, 
                    t):
        
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start + 
            extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        posterior_variance = extract_into_tensor(self.posterior_veriance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    

    def p_mean_variance(self, x, t, clip_denoised: bool):

        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)

        elif self.parameterization == "x0":
            x_recon = model_out

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, 
                                                                                  x_t=x,
                                                                                  t=t)
        
        return model_mean, posterior_variance, posterior_log_variance
    

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):

        b, *_, device = *x.shape, x.device 
        model_mean, _, model_log_variance = self.p_mean_variance(x, t, clip_denoised)
        noise = noise_like(shape=x.shape, device=device, repeat=repeat_noise)

        # no noise when t == 0 
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):

        device = self.betas.device 
        b = shape[0]
        img = torch.randn(shape, device)
        intermediates = [img]

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, 
                                t=torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised
                                )
            
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)

        if return_intermediates:
            return img, intermediates
        
        return img 
    

    @torch.no_grad()
    def smaple(self, batch_size=16, return_intermediates=False):

        image_size = self.image_size
        channels = self.channels

        return self.p_sample_loop(shape=(batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)
    


    def q_sample(self, x_start, t, noise=None):

        noise = default(noise, lambda: torch.rand_like(x_start))

        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + 
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
    

    def get_loss(self, pred, target, mean=True):

        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()


        elif self.loss_type == "l2":
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)

            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')


        else:
            raise NotImplementedError("Unknown loss type '{loss_type}' ")
        

        return loss 
    


    def p_losses(self, x_start, t, noise=None):

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise 

        elif self.parameterization == "x0":
            target = x_start

        else:
            raise NotImplementedError(f"Parametrization '{self.parameterization}' not yet supported.")
        

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])
        log_prefix = "train" if self.training else "val"

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        print(f"Let's check the what is input to get self.lvlb_weights: {self.lvlb_weights}")
        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict
    

    def forward(self, x, *args, **kwargs):
        
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()

        return self.p_losses(x, t, *args, **kwargs)
    

    def get_input(self, batch, k):

        x = batch[k]

        return x 
    

    def shared_steps(self, batch):

        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)

        return loss, loss_dict
    

    def training_step(self, batch, batch_idx):

        loss, loss_dict = self.shared_steps(batch)

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

        _, loss_dict_no_ema = self.shared_steps(batch)

        with self.ema_scope():
            _, loss_dict_ema = self.shared_steps(batch)
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
    

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):

        log = dict()
        x = self.get_input(batch, self.first_stage_key)

        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)

        x = x.to(self.device)[:N]
        log["inputs"] = x 


        # get diffusion row 
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()

                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

                diffusion_row.append(x_noisy)


        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoised row 
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.smaple(batch_size=N, return_intermediates=True)

            log["samples"] = samples 
            log["denoise_row"] = self._get_rows_from_list(denoise_row)


        if return_keys:
            
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log 
            
            else:
                return {key: log[key] for key in return_keys}
            
        return log 
    

    def configure_optimizers(self):
        
        lr = self.learning_rate 
        params = list(self.model.parameters())

        if self.learn_logvar:
            params = params + [self.logvar]

        opt = torch.optim.AdamW(params, lr=lr)

        return opt 
    


class LatentDiffusion(DDPM):

    """main class"""

    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 *args,
                 **kwargs):
        

        self.learning_rate = 1.0e-04 
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std

        assert self.num_timesteps_cond <= kwargs['timesteps']

        # for backwards compatibility after implementation of DiffusionWrapper 
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'

        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None 

        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])

        super().__init__(conditioning_key=conditioning_key,
                         *args,
                         **kwargs)
        
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

        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False 
        self.bbox_tokenizer = None 

        self.restarted_from_ckpt = False 
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True


    def make_cond_schedule(self):
        self.cond_ids = torch.full(size=(self.num_timesteps,),
                                   fill_value=self.num_timesteps - 1,
                                   dtype=torch.long)
        
        ids = torch.round(torch.linspace(0, self.num_timesteps -1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids 


    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        
        # This condition block ensures the logic only runs once, at the very first batch of training.
        # and only if 'scale_by_std' is enabled and the model is not being resumed from a checkpoint.
        if self.scale_by_std and \
            self.current_epoch == 0 and \
            self.global_step == 0 and \
            batch_idx == 0 and not \
            self.restarted_from_ckpt:

            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'

            # set rescale weight to 1./std of encodings 
            print('### USING STD-RESCALING')
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)

            encoder_posterior = self.encode_first_stage(x)
            # obtain the latent representation 'z' from the encoder_posterior and detach if 
            # from the computation graph to prevent gradients from flowing back to the encoder during this scaling factor calculation.
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor

            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f'Setting self.scale_factor to {self.scale_factor}')
            print(f"###Using Std-Scaling")


    
    def register_schedule(self, 
                          given_betas=None, 
                          beta_schedule="linear", 
                          timesteps=1000, 
                          linear_start=0.0001, 
                          linear_end=0.02, 
                          cosine_s=0.008):
        

        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1 
        if self.shorten_cond_schedule:
            self.make_cond_schedule()


    def instantiate_first_stage(self,  config):

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



    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):

        """
            Generate a grid of images from a list of denoised latent vectors.

            This function decodes a list of latent representation at different denoising steps 
            into images and arranges them in a grid layout where:
                - Each row show different denoising steps for the same image 
                - Each column show the same denoising steps for different images.
        """

        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                       force_not_quantize=force_no_decoder_quantization))
            

        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_steps, n_row, C, H, W 
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)

        return denoise_grid
    

    def get_first_stage_encoding(self, encoder_posterior):

        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()

        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior 

        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        
        return self.scale_factor * z 
    

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
    

    def meshgrid(self, h, w):

        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr 
    

    def delta_border(self, h, w):

        """ 
        THis function calculates the normalized distance of each pixel in an image to it's nearest 
        border.

        :param h: height 
        :param w: width 
        :return: normalized distance to image border,
            with min distance = 0 at border and max dist = 0.5 at image center
        """

        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner 

        dist_left_up = torch.min(arr, dim=1, keepdim=True)[0]
        dist_right_down = torch.min(1 - arr, 
                                    dim=-1,
                                    keepdim=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]

        return edge_dist
    

    def get_weighting(self, h, w, Ly, Lx, device):

        """ 
            Compute weights for image patches based on their distance from the image border.

            This function generates a weighting tensor for patches in an image, where the weights 
            are determined by the normalized distance of each pixel from the image border. The 
            weights are clipped to ensure they stay within specified bounds, are optionally 
            adjusted by a tie-breaker weighting based on patch positions.
        """

        weighting = self.delta_border(h, w)
        # clip the weights to ensure they are within the specified min and max bounds
        weighting = torch.clip(weighting, 
                               self.split_input_params["clip_min_weight"],
                               self.split_input_params["clip_max_weight"])
        # Reshape the weighting tensor to (1, h*w, 1) and repeat it to match the number of patches (Ly * Lx)
        weighting = weighting.view(1, h*w, 1).repeat(1, 1, Ly*Lx).to(device)

        # check if tie-breaker weighting is enabled
        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting,
                                     self.split_input_params["clip_min_the_weight"],
                                     self.split_input_params["clip_max_the_weight"])
            
            L_weighting = L_weighting.view(1, 1, Ly*Lx).to(device)
            weighting = weighting * L_weighting

        return weighting
    

    def get_fold_unfold(self,
                        x, 
                        kernel_size,
                        stride,
                        uf=1,
                        df=1):
        
        """ 
        Split an image into patches (unfold) and prepare tools to re-assemble patches (fold),
        Also computes normalization weights to handle overlapping regions during reassembly

        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """

        bs, nc, h, w = x.shape 

        # calculate number of patches in height (Ly) and width (Lx)
        Ly = (h - kernel_size[0]) // stride[0] + 1 
        Lx = (w - kernel_size[1]) // stride[1] + 1 

        # Case 1: No Scaling (uf=1, df=1)
        if uf == 1 and df == 1:

            # configure unfolding (patch extraction)
            fold_params = dict(kernel_size=kernel_size,
                               dilation=1,
                               padding=0, 
                               stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            # configure folding (patch reassembly)
            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            # create weighting mask for blending patches 
            weighting = self.get_weighting(kernel_size[0], 
                                           kernel_size[1],
                                           Ly,
                                           Lx,
                                           x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, 1) 
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))


        # Case 2: Upscaling 
        elif uf > 1 and df == 1:

            # Unfold with original kernel/stride 
            fold_params = dict(kernel_size=kernel_size,
                               dilation=1,
                               padding=0,
                               stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            # fold with upscaled kernel/stride 
            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                                dilation=1,
                                padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[2] * uf), **fold_params2)


            # weighting for upscale output 
            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, 
                                           Ly,
                                           Lx,
                                           x.device)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf) 
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        # Case 3: Downscaling 
        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, 
                               dilation=1,
                               padding=0,
                               stride=stride)
            
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                                dilation=1,
                                padding=0,
                                stride=(stride[0] // df, stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, 
                                           kernel_size[1] // df,
                                           Ly,
                                           Lx,
                                           x.device).to(x.dtype)
            
            normalization = fold(weighting).view(1, 1, h // df, w // df)
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))


        else:
            raise NotImplementedError
        

        return fold, unfold, normalization, weighting
    


    @torch.no_grad()
    def get_input(self, 
                  batch, 
                  k, 
                  return_first_stage_outputs=False,
                  force_c_encode=False,
                  cond_key=None,
                  return_original_cond=False,
                  bs=None):
        

        x = super().get_input(batch, k)
        if bs is not None:
            x = x[:bs]

        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key 

            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'coordinates_bbox', 'txt']:
                    xc = batch[cond_key]

                elif cond_key == 'class_lebel':
                    xc = batch 

                else:
                    xc = super().get_input(batch, cond_key).to(self.device)


            else:
                xc = x 


            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    c = self.get_learned_conditioning(xc)

                else:
                    c = self.get_learned_conditioning(xc.to(self.device))


            else:
                c = xc 


            if bs is not None:
                c = c[:bs]

            if self.use_postional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)

                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c,
                     'pos_x': pos_x,
                     'pos_y': pos_y}
                

        else:


            c = None 
            xc = None 
            if self.use_postional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x,
                     'pos_y': pos_y}
                

        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])

        if return_original_cond:
            out.append(xc)

        return out 
    

    @torch.no_grad()
    def decode_first_stage(self,
                           z, 
                           predict_cids=False,
                           force_not_quantize=False):
        
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()

            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z 

        if hasattr(self, "split_input_params"):

            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]
                stride = self.split_input_params["stride"]
                uf = self.split_input_params["vqf"]

                bs, nc, h, w = z.shape 
                if ks[0] > h or ks[1] > w:
                    ks = (min[ks[0], h], min(ks[1], w))
                    print("Reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("Reducing Stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)
                # 1. reshape the image shape 
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))

                # 2. apply model loop over last dim 
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                                                 
                                    for i in range(z.shape[-1])]
                    
                else:
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]
                    

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse stage 1. reshape to img shape 
                o = o.view((o.shape[0], -1, o.shape[-1]))

                decoded = fold(o)
                decoded = decoded / normalization 
                return decoded
            

            else:
                
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                
                else:
                    return self.first_stage_model.decode(z)
                

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            
            else:
                return self.first_stage_model.decode(z)
            

    # same as above but without decorator 
    def differentiable_decode_first_stage(self,
                           z, 
                           predict_cids=False,
                           force_not_quantize=False):
        
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()

            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z 

        if hasattr(self, "split_input_params"):

            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]
                stride = self.split_input_params["stride"]
                uf = self.split_input_params["vqf"]

                bs, nc, h, w = z.shape 
                if ks[0] > h or ks[1] > w:
                    ks = (min[ks[0], h], min(ks[1], w))
                    print("Reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("Reducing Stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)
                # 1. reshape the image shape 
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))

                # 2. apply model loop over last dim 
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                                                 
                                    for i in range(z.shape[-1])]
                    
                else:
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]
                    

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse stage 1. reshape to img shape 
                o = o.view((o.shape[0], -1, o.shape[-1]))

                decoded = fold(o)
                decoded = decoded / normalization 
                return decoded
            

            else:
                
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                
                else:
                    return self.first_stage_model.decode(z)
                

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            
            else:
                return self.first_stage_model.decode(z)
            

    @torch.no_grad()
    def encode_first_stage(self, x):

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:

                ks = self.split_input_params["ks"]
                stride = self.split_input_params["stride"]
                df = self.split_input_params["vqf"]

                self.split_input_params["original_image_size"] = x.shape[-2:]
                bs, nc, h, w = x.shape 

                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("Reducing the kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("Reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)

                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))

                output_list = [self.first_stage_model.encode(z[:, :, :, :, i])
                               for i in range(z.shape[-1])]
                
                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape 
                o = o.view((o.shape[0], -1, o.shape[-1]))
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded
            

            else:
                return self.first_stage_model.encode(x)
            
        else:
            return self.first_stage_model.encode(x)
        

    def shared_steps(self, batch, **kwargs):
        
        x, c = self.get_input(batch, self.first_stage_key)
        loss = self(x, c)
        return loss 
    

    def forward(self, x, c, *args, **kwargs):

        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()

        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)

            if self.shorten_cond_schedule:
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, 
                                  t=tc,
                                  noise=torch.randn_like(c.float()))
                
        return self.p_losses(x, c, t, *args, **kwargs)
    


    def _rescale_annotations(self,
                             bboxes,
                             crop_coordinates):
        
        """Rescaling bounding box annotations to normalized coordinates (0-1) relative to a crop region."""
        
        def rescale_bbox(bbox):
            x0 = torch.clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2])
            y0 = torch.clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3])

            w = min(bbox[2] / crop_coordinates[2], 1 - x0)
            h = min(bbox[3] / crop_coordinates[3], 1 - y0)

            return x0, y0, w, h 
        
        return [rescale_bbox(b) for b in bboxes]
    

    def apply_model(self,
                    x_noisy, 
                    t, 
                    cond, 
                    return_ids=False):
        
        if isinstance(cond, dict):
            pass 

        else:

            if isinstance(cond, list):
                cond = [cond]

            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}


        if hasattr(self, 'split_input_params'):

            assert len(cond) == 1 
            assert not return_ids

            ks = self.split_input_params["ks"]
            stride = self.split_input_params["stride"]

            h, w = x_noisy.shape[-2:]
            fold, unfold, normalization, weighting = self.get_fold_unfold(x_noisy, ks, stride)

            z = unfold(x_noisy)
            z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]

            if self.cond_stage_key in ["image", "LR_image", "segmentation", "bbox_img"] and self.model.conditioning_key:

                c_key = next(iter(cond.keys()))
                c = next(iter(cond.values()))

                assert (len(c) == 1)
                c = c[0]

                c = unfold(c)
                c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))
                cond_list = [{c_key: [c[:, :, :, :, i]]} for i in range(c.shape[-1])]


            elif self.cond_stage_key == 'coordinates_bbox':

                assert 'original_image_size' in self.split_input_params, "BoundingBoxRescaling is missing original_image_size"

                # assuming padding of unfold is always 0 and its dilation is always 1 
                n_patches_per_row = int((w - ks[0]) / stride[0] + 1)
                full_img_h, full_img_w = self.split_input_params["original_image_size"]

                # as we are operating on latent, we need the factor from the original image size to the 
                # spatial latent size to properly rescale the crops for regenerting the bbox annotations 
                num_downs = self.first_stage_model.encoder.num_resolutions - 1 
                rescale_latent = 2 ** (num_downs)

                # get top left positions of patches as conforming for the bbox tokenizer, therefore we 
                # need to rescale the t1 patch coordinates to be in between (0, 1)
                t1_patch_coordinates = [(rescale_latent * stride[0] * (patch_nr % n_patches_per_row) / full_img_w,
                                         rescale_latent * stride[1] * (patch_nr // n_patches_per_row) / full_img_h)
                                         for patch_nr in range(z.shape[-1])]
                

                # patch_limits are t1_coord, with and height coordinates as (x_t1, y_t1, h, w)
                patch_limits = [(x_t1, y_t1,
                                 rescale_latent * ks[0] / full_img_w,
                                 rescale_latent * ks[1] / full_img_h) for x_t1, y_t1 in t1_patch_coordinates]
                
                
                # tokenize crop coordinates for the bounding boxes of the respective patches 
                patch_limits_tknzd = [torch.LongTensor(self.bbox_tokenizer.__crop_encoder(bbox))[None].to(self.device) 
                                      for bbox in patch_limits]
                
                print(patch_limits_tknzd[0].shape)

                # cut tknzd crop position from conditiong 
                assert isinstance(cond, dict), 'cond must be dict to be fed into model'
                cut_cond = cond['c_crossattn'][0][..., :-2].to(self.device)
                print(cut_cond.shape)

                adapted_cond = torch.stack([torch.cat([cut_cond, p], dim=1) for p in patch_limits_tknzd])
                adapted_cond = rearrange(adapted_cond, 'l b n -> (l b) n')
                print(adapted_cond.shape)

                adapted_cond = self.get_learned_conditioning(adapted_cond)
                print(adapted_cond.shape)

                adapted_cond = rearrange(adapted_cond, '(l b) n d -> l b n d', l=z.shape[-1])
                print(adapted_cond.shape)

                cond_list = [{'c_crossattn': [e]} for e in adapted_cond]

            else:

                cond_list = [cond for i in range(z.shape[-1])]

            # apply model by loop over crop 
            output_list = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
            
            assert not isinstance(output_list[0],
                                  tuple)
            

            o = torch.stack(output_list, axis=-1)
            o = o * weighting

            # Reversing reshape to img shape 
            o = o.view((o.shape[0], -1, o.shape[-1]))
            x_recon = fold(o) / normalization

        else:

            x_recon = self.model(x_noisy, t, **cond)

        
        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[o]
        
        else:
            return x_recon
        


    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):

        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    


    def _prior_bpd(self, x_start):

        """ 
        Get the prior KL term for the variational lower-bound measured in 
        bits-per-dim 

        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """


        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)

        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean,
                             logvar1=qt_log_variance,
                             mean2=0.0,
                             logvar2=0.0)
        
        return mean_flat(kl_prior) / np.log(2.0)
    

    def p_losses(self, x_start, cond, t, noise=None):
        
        noise = default(noise, lambda: torch.rand_like(x_start))
        x_noisy = self.q_sample(x_start=x_start,
                                t=t,
                                noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start

        elif self.parameterization == "eps":
            target = noise

        else:
            raise NotImplementedError()
        

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})


        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t

        if self.learn_logvar:
            loss_dict.update({f"{prefix}/loss_gamma": loss.mean()})
            loss_dict.update({"logvar": self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weight[t] * loss_vlb)
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict
    


    def p_mean_variance(self, 
                        x, 
                        c,
                        t, 
                        clip_denoised: bool,
                        return_codebook_ids=False,
                        quantize_denoised=False,
                        return_x0=False,
                        score_corrector=None,
                        corrector_kwargs=None):
        
        

        t_in = t 
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is  not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)


        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, 
                                                    t=t,
                                                    noise=model_out)
            
        elif self.parameterization == "x0":
            x_recon = model_out

        else:
            raise NotImplementedError()
        

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon,
                                                                                  x_t=x,
                                                                                  t=t)
        
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        
        else:
            return model_mean, posterior_variance, posterior_log_variance
        

    
    @torch.no_grad()
    def p_sample(self, 
                 x, 
                 c,
                 t,
                 clip_denoised=False,
                 repeat_noise=False,
                 return_codebook_ids=False,
                 quantize_denoised=False,
                 return_x0=False,
                 temperture=1.,
                 noise_dropout=0.,
                 score_corrector=None,
                 corrector_kwargs=None):
        
        b, *_, device = *x.shape, x.device 
        outputs = self.p_mean_variance(x=x,
                                       c=c,
                                       t=t,
                                       clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector,
                                       corrector_kwargs=corrector_kwargs)
        
        if return_codebook_ids:
            raise DeprecationWarning("Supporting dropped.")
            model_mean, _, model_log_variance, logits = outputs

        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs

        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperture

        if noise_dropout > 0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)

        # no noise when t == 0 
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,)) * (len(x.shape) - 1))

        if return_codebook_ids:
            return model_mean * nonzero_mask * (0.5 * model_log_variance).exp() * noise, torch.logits.argmax(dim=1)
        
        if return_x0:
            return model_mean * nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        


    @torch.no_grad()
    def progressive_denoising(self,
                              cond,
                              shape,
                              verbose=True,
                              callback=None,
                              quantize_denoised=False,
                              img_callback=None,
                              mask=None,
                              x0=None,
                              temperture=1.,
                              noise_dropout=0.,
                              score_corrector=None,
                              corrector_kwargs=None,
                              batch_size=None,
                              x_T=None,
                              start_T=None,
                              log_every_t=None
                              ):
        



        if not log_every_t:
            log_every_t = self.log_every_t

        timesteps = self.num_timesteps

        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)

        else:
            b = batch_size = shape[0]

        if x_T is None:
            img = torch.randn(shape, device=self.device)

        else:
            img = x_T

        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else 
                        list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
                
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]



        if start_T is not None:
            timesteps = min(timesteps, start_T)

        iterator = tqdm(reversed(range(0, timesteps)), desc="Progressive Generator",
                        total=timesteps) if verbose else reversed(
                            range(0, timesteps)
                        )
        
        if type(temperture) == float:
            temperture = [temperture] * timesteps


        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)

            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)

                cond = self.q_sample(x_start=cond,
                                     t=tc,
                                     noise=torch.rand_like(cond))
                
            img, x0_partial = self.p_sample(x=img,
                                            c=cond,
                                            t=ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised,
                                            return_x0=True,
                                            temperture=temperture[i],
                                            noise_dropout=noise_dropout,
                                            score_corrector=score_corrector,
                                            corrector_kwargs=corrector_kwargs)
            

            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)

                img = img_orig * mask + (1. - mask) * img 

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)

            if callback: callback(i)
            if img_callback: img_callback(img, i)

        return img, intermediates
    



    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None,**kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)

    @torch.no_grad()
    def sample_log(self,cond,batch_size,ddim, ddim_steps,**kwargs):

        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates =ddim_sampler.sample(ddim_steps,batch_size,
                                                        shape,cond,verbose=False,**kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True,**kwargs)

        return samples, intermediates


    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, **kwargs):

        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"])
                log["conditioning"] = xc
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                         ddim_steps=ddim_steps,eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                # also display when quantizing x0 while sampling
                with self.ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                             ddim_steps=ddim_steps,eta=ddim_eta,
                                                             quantize_denoised=True)
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
                    #                                      quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

            if inpaint:
                # make a simple center square
                b, h, w = z.shape[0], z.shape[2], z.shape[3]
                mask = torch.ones(N, h, w).to(self.device)
                # zeros will be filled in
                mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
                mask = mask[:, None, ...]
                with self.ema_scope("Plotting Inpaint"):

                    samples, _ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim, eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_inpainting"] = x_samples
                log["mask"] = mask

                # outpaint
                with self.ema_scope("Plotting Outpaint"):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_outpainting"] = x_samples

        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log
    


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
            assert 'target' in self.schedule_config 
            scheduler = instantiate_from_config(self.schedule_config)
            print("setting up LambdaLR scheduler...")

            scheduler = [
                {
                    "scheduler": LambdaLR(optimizer=opt,
                                          lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1
                }
            ]
            return [opt], scheduler
        
        return opt 
    


    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x
    







class DiffusionWrapper(pl.LightningModule):

    def __init__(self,
                 diff_model_config,
                 conditioning_key):
        
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']


    def forward(self, 
                x, 
                t,
                c_concat: list=None,
                c_crossattn: list = None):
        

        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)

        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)

        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)

        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)

        elif self.conditioning_key == "adm":
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)

        else:
            raise NotImplementedError()
        
        return out 
    



    
def disabled_train(self, mode=True):
        
    return self 





__conditioning_keys__ = {
    'concat': 'c_concat',
    'crossattn': 'c_crossattn',
    'adm': 'y'
}