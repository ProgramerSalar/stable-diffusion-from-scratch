import torch.nn as nn 
import torch 
import numpy as np 
from einops import rearrange

class VectorQuantize2(nn.Module):
    

    """
    Improved version of VectorQuantize that can be used as a drop-in replacement.
    Optimizers performance by avoiding costly matrix multiplications and allows for 
    post-hoc remapping of indices.

    Args:
        n_e (int): Number of embedding vectors (codebook size)
        e_dim (int): Dimension of embedding vectors 
        beta (float): Commitment cost weighting factor 
        remap (str, optional): Path to remapping file for indices 
        unknown_index (str or int, optional): How to handle unknown indices ("random", "extra", or integer)
        sane_index_shape (bool): Whether to reshape indices to match input spatial dimensions 
        legacy (bool): Whether to use legacy loss computation

    """


    def __init__(self, 
                 n_e,
                 e_dim,
                 beta,
                 remap=None,
                 unknown_index="random",
                 sane_index_shape=False,
                 legacy=True):
        
        super().__init__()
        self.n_e = n_e 
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        # create embedding layer 
        self.embedding = nn.Embedding(num_embeddings=self.n_e, 
                                      embedding_dim=self.e_dim)
        
        # Handle index remapping if specified 
        self.remap = remap
        if self.remap is not None:
            # Load precomputed used indices 
            self.register_buffer("used", tensor=torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]  # Number of used indices 
            self.unknown_index = unknown_index  # how to handle unknown indices 

            # Handle 'extra' option for unknown indices 
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1 

            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. " 
                  f"Using {self.unknown_index} for unknown indices.")
            

        else:
            self.re_embed = n_e  # No remapping, use all indices 

        self.sane_index_shape = sane_index_shape   

    def remap_to_used(self, inds):
        """ 
        Remap indices to only used indices in codebook.

        Args:
            inds (torch.Tensor): Original indices to  remap 

        Returns:
            torch.Tensor: Remapped indices
        """


        ishape = inds.shape 
        assert len(ishape) > 1
        # Flatten spatial dimension 
        inds = inds.reshape(ishape[0], -1) 
        # Get used indices 
        used = self.used.to(inds)

        # Find matches between input indices and used indices 
        match = (inds[:, :, None] == used[None, None, ...]).long()
        # Get best matching used index 
        new = match.argmax(-1)
        # Identify unknown indices 
        unknown = match.sum(2)<1

        # Handle unknown indices 
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)

        else:
            # Assign specified unknown index 
            new[unknown] = self.unknown_index

        return new.reshape(ishape)


    def forward(self, 
                z,
                temp=None,
                rescale_logits=False,
                return_logits=False):
        
        """ 
        Forward pass for vector quantization.

        Args:
            z (torch.Tesnsor): Input tensor to quantize 
            temp (float, optional): Temperature for Gumbel softmax (unused)
            rescale_logits (bool, optional): Whether to rescale logits (unused)
            return_logits (bool, optional): Whether to return logits (unused)

        Returns:
            tuple: (quantized output, loss, (perplexity, min_encoding, min_encoding, indices))
        """


        # Interface compatibility checks 
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"

        print("Z: ", z)

        # Rearrange input to (batch, height, width, channels)
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        # Flatten spatial dimensions
        z_flattened = z.view(-1, self.e_dim)

        # Compute distances between input vectors and codebook vectors 
        # Using (x-y)^2 = x^2 + y^2 - 2xy expansion for efficiency 
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn -> bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
        
        # Find closest codebook vectors
        min_encoding_indices = torch.argmin(input=d, dim=1)
        # Get quantized vectors 
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None 
        min_encoding = None 

        # compute loss for embedding 
        if not self.legacy:
            loss = self.beta * torch.mean(input=(z_q.detach()-z)**2) + \
                    torch.mean(input=(z_q - z.detach()) ** 2)

        else:
            loss = torch.mean(input=torch.mean(z_q.detach()-z) ** 2) + self.beta * \
                    torch.mean((z_q - z.detach()) ** 2)
            

        # preserve gradients using straight-through estimator 
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape (batch, channels, height, width)
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()


        # Remap indices if specified 
        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1) 
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)


        # Reshape indices to match spatial dimensions if requrested
        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], 
                z_q.shape[2],
                z_q.shape[3]
            )

        return z_q, loss, (perplexity, min_encoding, min_encoding_indices)
    


if __name__ == "__main__":

    # configuration parameters 
    batch_size = 4 
    channels = 64 
    height = 16 
    width = 16 
    n_emeddings = 512    # Codebook size 
    embedding_dim = 64    # Same as channels 
    beta = 0.25         # Commitment cost 


    # create a sample input tensor (could be output from a CNN encoder)
    z = torch.randn(batch_size, channels, height, width)
# -----------------------------------------------------------------------------------------------------------------------------
    # # Initialize the vector quantizer 
    # vq = VectorQuantize2(
    #     n_e=n_emeddings,
    #     e_dim = embedding_dim,
    #     beta=beta,
    #     remap=None,   # Could provide path to .npy file with used indices 
    #     unknown_index="random",
    #     sane_index_shape=True,
    #     legacy=False
    # )

    # # print(vq)

    # # Forward pass 
    # z_q, loss, (perplexity, min_encodings, min_encoding_indices) = vq(z)

    # # print results 
    # print("Input shape: ", z.shape)
    # print("Quantized output shape: ", z_q.shape)
    # print("Loss: ", loss.item())
    # print("Indices shape: ",  min_encoding_indices.shape)

    # print("\nFirst few indices: ")
    # print(min_encoding_indices[0, :5, :5])

    # # verify the output make sense 
    # print("\nVerification")
    # print("Input norm: ", torch.norm(z).item())
    # print("Quantized norm: ", torch.norm(z_q).item())
    # print("Difference norm: ", torch.norm(z_q - z).item())
# -------------------------------------------------------------------------------------------------
    # Example of remapping functionality (pretend some indices are unused)
    if True:
        print("\n Testing remapping...")

        # Create a fake remapping file (normally you had load a real one)
        used_indices = np.array([10, 20, 30, 40, 50])
        np.save("used_indices.npy", used_indices)

        # Reinintialize with remapping 
        vq_remap = VectorQuantize2(
            n_e=n_emeddings,
            e_dim=embedding_dim,
            beta=beta,
            remap="used_indices.npy",
            unknown_index="random",
            sane_index_shape=True,
            legacy=False
        )

        # # Forward pass 
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = vq_remap(z)

        # Forward pass with remapping 
        _, _, (_, _, remapped_indices) = vq_remap(z)
        print("Original indices sample: ", min_encoding_indices[0, 0, :5]) 
        print("Remapped indices sample: ", remapped_indices[0, 0, :5])
        print("Max original index: ", min_encoding_indices.max().item())
        print("Max remapped index: ", remapped_indices.max().item())






