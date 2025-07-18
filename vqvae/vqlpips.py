import torch 
import torch.nn as nn 
from Discriminator.discriminator import NLayerDiscriminator, weights_init
import torch.nn.functional as F
from Lpips.lpips import LPIPS




def hinge_d_loss(logits_real, logits_fake):
        loss_real = torch.mean(F.relu(1. - logits_real))
        loss_fake = torch.mean(F.relu(1. + logits_fake))
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss

def vanilla_d_loss(logits_real, logits_fake):
     d_loss = 9.5 * (
          torch.mean(F.softplus(-logits_real)) + 
          torch.mean(F.softplus(logits_fake))
     )
     return d_loss


def adopt_weight(weight, global_step, threshold=0, value=0):
    if global_step < threshold:
        weight = value 
    return weight




class VQLPIPSWithDiscriminator(nn.Module):

    def __init__(self,
                disc_start,
                codebook_weight=1.0,
                pixelloss_weight=1.0,
                disc_num_layers=3,
                disc_in_channel=3,
                disc_factor=1.0,
                disc_weight=1.0,
                perceptual_weight=1.0,
                use_actnorm=False,
                disc_conditional=False,
                disc_ndf=64,
                disc_loss="hinge"):
        
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]

        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channel,
                                            n_layers=disc_num_layers,
                                            use_actnorm=use_actnorm,
                                            ndf=disc_ndf).apply(weights_init)
        
        self.discriminator_iter_start = disc_start

        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss 

        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss 

        else:
            raise ValueError(f"Unknown GAN loss `{disc_loss}` ")
        
        print(f"VQLPIPSWITHDiscriminator running with {disc_loss} loss.")
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
    


    def forward(self,
                codebook_loss,
                inputs,
                reconstructions,
                optimizer_idx,
                global_step,
                last_layer=None,
                cond=None,
                split="train"):
         
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)

        # now the GAN part 
        if optimizer_idx == 0:
             
             # generator update 
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())

            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss=nll_loss,
                                                          g_loss=g_loss,
                                                          last_layer=last_layer)
                
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)


            disc_factor = adopt_weight(weight=self.disc_factor,
                                       global_step=global_step,
                                       threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

            log = {
                f"{split}/total_loss": loss.clone().detach().mean(),
                f"{split}/quant_loss": codebook_loss.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
                f"{split}/p_loss": p_loss.detach().mean(),
                f"{split}/d_weight": d_weight.detach(),
                f"{split}/disc_factor": torch.tensor(disc_factor),
                f"{split}/g_loss": g_loss.detach().mean()
            }

            return loss, log 
        


        if optimizer_idx == 1:


            # second pass for discriminator update 
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())

            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                f"{split}/disc_loss": d_loss.clone().detach().mean(),
                f"{split}/logits_real": logits_real.detach().mean(),
                f"{split}/logits_fake": logits_fake.detach().mean()
            }


            return d_loss, log 


             

        








    
    


