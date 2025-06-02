import torch 
from torch import nn 
from Lpips.lpips import LPIPS
from Discriminator.discriminator import NLayerDiscriminator, weights_init
from torch.nn import functional as F

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

class LPIPSWithDiscriminator(nn.Module):

    def __init__(self,
                 disc_start,                # step at which to start using the discriminator
                 logvar_init=0.0,           # initial value for the learnable log variance
                 kl_weight=1.0,             # weight for KL divergance loss 
                 pixelloss_weight=1.0,      # weight for pixel-wise L1 loss
                 disc_num_layers=3,         # number of layers in discriminator
                 disc_in_channels=3,        # numbef of input channels for discriminator
                 disc_factor=1.0,           # weight factor for discriminator loss
                 disc_weight=1.0,           # weight for discriminator loss
                 perceptual_weight=1.0,     # weight for LPIPS perceptual loss
                 use_actnorm=False,         # whether to use activation normalization in discriminator 
                 disc_conditional=False,    # whether discriminator is conditional
                 disc_loss="hinge"          # type of discriminator loss ('hinge' or 'vanilla')
                 ):
        

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        # output log variance 
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm).apply(weights_init)
        

        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional


    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        
        # computes L1 reconstruction loss
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())

        # Adds Lpips perceptual loss if weight > 0
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        # computes negative log-liklihood loss with learned variance
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar

        # Applies optional per-pixel weights and averages losses
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        # computes and averages KL divergance loss 
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:

            # Get's discriminator predictions for reconstructed images
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))

            # Generator loss (maximize discriminator fake classification)
            g_loss = -torch.mean(logits_fake)

            # computes adaptive weight or default to 0
            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            
            # computes total loss with scheduled weighting 
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            # Ruturns loss and logging dictionary
            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:

            # Gets discriminator predictions for real and fake images 
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            # computes discriminator loss with scheduled weighting 
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            # Return discriminator loss and logging dictionary
            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log



if __name__ == "__main__":

    import torch
    from torch import nn
    from torch.distributions import Normal, kl_divergence

    from torch.distributions import register_kl

    

    
    

    class DiagonalGaussianDistribution:
        def __init__(self, mean, logvar):
            self.mean = mean
            self.logvar = logvar
        
        def kl(self):
            return kl_divergence(
                self,
                Normal(torch.zeros_like(self.mean), 
                torch.ones_like(self.logvar).exp()
            ))


    @register_kl(DiagonalGaussianDistribution, Normal)
    def kl_diagonal_normal(p, q):
        return 0.5 * (torch.exp(p.logvar) + p.mean**2 - 1 - p.logvar)




    from torch.utils.data import DataLoader
    

    real_images = torch.randn(4, 3, 256, 256).cuda()
    fake_images = torch.randn(4, 3, 256, 256).cuda()

    mean = torch.randn(4, 128, 32, 32).cuda()
    logvar = torch.randn(4, 128, 32, 32).cuda()



    posteriors = DiagonalGaussianDistribution(mean, logvar)   # Dummy posterior (for KL loss)

    # Initialize the loss module 
    loss_fn = LPIPSWithDiscriminator(
        disc_start=5000,    # start discriminator after 5000 steps 
        disc_in_channels=3,    # RGB input
        disc_weight=1.0,         # Weight for discriminator loss 
        disc_loss="hinge",
        perceptual_weight=1.0,   # Lpips weight 
        kl_weight=1.0           # KL divergance weight
    ).cuda()

    # optimizers (one for generator, one for discriminator)
    opt_g = torch.optim.Adam(loss_fn.parameters(), lr=1e-4)
    opt_d = torch.optim.Adam(loss_fn.discriminator.parameters(), lr=1e-4)


    for global_step in range(10000):

        # Generator Update (optimizer_idx=0)
        opt_g.zero_grad()
        loss_g, log_g = loss_fn(
            inputs=real_images,
            reconstructions = fake_images,
            posteriors = posteriors,
            optimizer_idx=0,
            global_step=global_step
        )
        loss_g.backward()
        opt_g.step()

        # Discriminator update 
        loss_d, log_d = loss_fn(
            inputs=real_images,
            reconstructions=fake_images,
            posteriors=posteriors,
            optimizer_idx=1,
            global_step=global_step
        )
        loss_d.backward()
        opt_d.step()



        # Logging 
        if global_step % 100 == 0:
            print(f"Step {global_step}: ")
            print(f"    \n Generator Loss: {log_g['train/total_loss'].item():.3f}")
            print(f"    \n Discriminator Loss: {log_d['train/disc_loss'].item():.3f}")
            print(f"    \n Logits Real: {log_d['train/logits_real'].item():.3f}")
            print(f"    \n Logits Fake {log_d['train/logits_fake'].item():.3f}")
            
    

    