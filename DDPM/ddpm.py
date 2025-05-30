import torch
import numpy as np

from tqdm import tqdm



def broadcast(values, broadcast_to):
    values = values.flatten()

    while len(values.shape) < len(broadcast_to.shape):
        values = values.unsqueeze(-1)

    return values


class DDPMPipeline:
    def __init__(self, beta_start=1e-4, beta_end=1e-2, num_timesteps=1000):
        # Betas settings are in section 4 of https://arxiv.org/pdf/2006.11239.pdf
        # Implemented linear schedule for now, cosine works better tho.
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas

        # alpha-hat in the paper, precompute them
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)

        self.num_timesteps = num_timesteps

    def forward_diffusion(self, images, timesteps) -> tuple[torch.Tensor, torch.Tensor]:
        """
        https://arxiv.org/pdf/2006.11239.pdf, equation (14), the term inside epsilon_theta
        :return:
        """
        gaussian_noise = torch.randn(images.shape).to(images.device)
        
        # Move self.alphas_hat to the same device as images 
        self.alphas_hat = self.alphas_hat.to(images.device)
        # Move timesteps to the same device as images
        timesteps = timesteps.to(images.device)

        # Now indexing will work 
        alpha_hat = self.alphas_hat[timesteps]
        alpha_hat = broadcast(alpha_hat, images)
        


        return torch.sqrt(alpha_hat) * images + torch.sqrt(1 - alpha_hat) * gaussian_noise, gaussian_noise

    def reverse_diffusion(self, model, noisy_images, timesteps):
        predicted_noise = model(noisy_images, timesteps)
        return predicted_noise

    @torch.no_grad()
    def sampling(self, model, initial_noise, device, save_all_steps=False):
        """
        Algorithm 2 from the paper https://arxiv.org/pdf/2006.11239.pdf
        Seems like we have two variations of sampling algorithm: iterative and with reparametrization trick (equation 15)
        Iterative assumes you have to denoise image step-by-step on T=1000 timestamps, while the second approach lets us
        calculate x_0 approximation constantly without gradually denosing x_T till x_0.

        :param model:
        :param initial_noise:
        :param device:
        :param save_all_steps:
        :return:
        """
        image = initial_noise
        images = []
        for timestep in tqdm(range(self.num_timesteps - 1, -1, -1)):
            # Broadcast timestep for batch_size
            ts = timestep * torch.ones(image.shape[0], dtype=torch.long, device=device)
            predicted_noise = model(image, ts)
            beta_t = self.betas[timestep].to(device)
            alpha_t = self.alphas[timestep].to(device)
            alpha_hat = self.alphas_hat[timestep].to(device)

            # Algorithm 2, step 4: calculate x_{t-1} with alphas and variance.
            # Since paper says we can use fixed variance (section 3.2, in the beginning),
            # we will calculate the one which assumes we have x0 deterministically set to one point.
            alpha_hat_prev = self.alphas_hat[timestep - 1].to(device)
            beta_t_hat = (1 - alpha_hat_prev) / (1 - alpha_hat) * beta_t
            variance = torch.sqrt(beta_t_hat) * torch.randn(image.shape).to(device) if timestep > 0 else 0

            image = torch.pow(alpha_t, -0.5) * (image -
                                                beta_t / torch.sqrt((1 - alpha_hat_prev)) *
                                                predicted_noise) + variance
            if save_all_steps:
                images.append(image.cpu())
        return images if save_all_steps else image
