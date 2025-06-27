import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel
import kornia

from clip_encoder.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test





class AbstractEncoder(nn.Module):

    def __init__(self):
        super().__init__()


    def encoder(self, *args, **kwargs):
        raise NotImplementedError
    

class ClassEmbedder(nn.Module):

    def __init__(self,
                 embed_dim,
                 n_classes=1000,
                 key="class"):
        
        super().__init__()
        self.key = key 
        self.embedding = nn.Embedding(n_classes, embed_dim)


    def forward(self, batch, key=None):
        if key is None:
            key = self.key 

        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)

        return c 
    

class TransformerEmbedder(AbstractEncoder):

    """Some transformer encoder layers"""

    def __init__(self,
                 n_embed,
                 n_layer,
                 vocab_size,
                 max_seq_len=77,
                 device="cuda"):
        
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size,
                                              max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))
        

    def forward(self, tokens):

        tokens = tokens.to(self.device)
        z = self.transformer(tokens, return_embeddings=True)
        return z 
    
    def encode(self, x):
        return self(x)
    

class BERTTokenizer(AbstractEncoder):

    """Uses a pretrained BERT tokenizer by huggingface. vocab size: 30522 (?)"""

    def __init__(self,
                 device="cuda",
                 vq_interface=True,
                 max_length=77):
        
        super().__init__()

        from transformers import BertTokenizerFast
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length


    def forward(self, 
                text):
        
        batch_encoding = self.tokenizer(text,
                                        truncation=True,
                                        max_length=self.max_length,
                                        return_length=True,
                                        return_overflowing_tokens=False,
                                        padding="max_length",
                                        return_tensors="pt")
        
        tokens = batch_encoding["input_ids"].to(self.device)

        return tokens
    

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        
        return None, None, [None, None, tokens]
    

    def decode(self, text):
        return text
    

class BERTEmbedder(AbstractEncoder):

    """Uses the BERT tokenizer model and add some transformer encoder layers"""

    def __init__(self,
                 n_embed,
                 n_layer,
                 vocab_size=30522,
                 max_seq_len=77,
                 device="cuda",
                 use_tokenizer = True,
                 embedding_dropout=0.0):
        
        super().__init__()

        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False,
                                         max_length=max_seq_len)
            
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size,
                                              max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)
        


    def forward(self, text):

        if self.use_tknz_fn:
            tokens = self.tknz_fn(text).cuda()

        else:
            tokens = text 

        z = self.transformer(tokens, return_embeddings=True)

        return z 
    

    def encode(self, text):
        return self(text)
    

class SpatialRescaler(nn.Module):

    def __init__(self,
                 n_stages=1,
                 method="bilinear",
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        
        super().__init__()
        self.n_stages = n_stages

        assert self.n_stages >= 0
        assert method in ["nearest", "linear", "bilinear", "trilinear", "bicubic", "area"]

        self.multiplier = multiplier
        self.interpolator = partial(nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None

        if self.remap_output:
            print(f"Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.")
            self.channel_mapper = nn.Conv2d(in_channels,
                                            out_channels,
                                            1,
                                            bias=bias)
            
        

    def forward(self, x):
        
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.remap_output:
          x = self.channel_mapper(x)

        return x 


    def encode(self, x):
        return self(x)


class FrozenCLIPEmbedder(AbstractEncoder):

    """Uses the CLIP transformer encoder for text (from Huggingface)"""

    def __init__(self,
                 version="openai/clip-vit-large-patch14",
                 device="cuda",
                 max_length=77):
        
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):

        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False 

    def forward(self, text):

        print(f"what is the input data: >>> {text}")
        print(f"what is the input data: >>> {text.shape}")

        batch_encoding = self.tokenizer(text,
                                        truncation=True,
                                        max_length=self.max_length,
                                        return_length=True,
                                        return_overflowing_tokens=False,
                                        padding="max_length",
                                        return_tensors="pt")
        

        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state 
        return z 
    

    def encode(self, text):
        return self(text)
    


class FrozenClipImageEmbedder(nn.Module):

    """Uses the CLIP image encoder."""

    def __init__(
            self,
            model,
            jit=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
            antialias=False
    ):
        
        super().__init__()
        self.model, _ = clip.load(name=model,
                                 device=device,
                                 jit=jit)
        
        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)


    
    def preprocess(self, x):

        # normalize to [0, 1]
        x = kornia.geometry.resize(x,
                                   (224, 224),
                                   interpolation="bicubic",
                                   align_corners=True,
                                   antialias=self.antialias)
        
        x = (x + 1.) / 2 
        # renormalize according to clip 
        x = kornia.enhance.normalize(x, self.mean, self.std)

        return x 

    def forward(self, x):

        # x is assumed to be in range [-1, 1]
        return self.model.encode_image(self.preprocess(x))
    




    

if __name__ == "__main__":
    

            #### ClassEmbedder
    # batch = {'class': torch.tensor([42, 127, 899])}
    # output = ClassEmbedder(embed_dim=512)
    # output = output(batch)
    # print(output.shape)

# ---------------------------------------------------------------------

            ### TransformerEmbedder
    # Create sample tokens (batch of token sequences)
    # Shape: (batch_size, sequence_length)
    # Tokens must be integers in range [0, vocab_size-1]
    # tokens = torch.tensor([
    #     [101, 2056, 2023, 2003, 1037, 2307, 9999, 0, 0, 0],  # Sequence 1 [CLS]  photo  of   a     cat   [UNK] [PAD] [PAD] ...
    #     [101, 4862, 2028, 1997, 1037, 4553, 9999, 0, 0, 0],   # Sequence 2
    #     [101, 1037, 2214, 2003, 1037, 2307, 9999, 0, 0, 0]    # Sequence 3
    # ], device="cuda:0")

    # # Pad to max_seq_len (77) - normally you'd use proper padding
    # padded_tokens = torch.nn.functional.pad(
    #     tokens, 
    #     (0, 77 - tokens.shape[1]), 
    #     value=0
    # )

    # print("Input tokens shape:", padded_tokens.shape)  

    # te = TransformerEmbedder(n_embed=512,
    #                          n_layer=6,
    #                          vocab_size=10000
    #                          ).cuda()
    
    # output = te(padded_tokens)
    # print(output.shape)

# ----------------------------------------------------------------


            ### BERTTOkenizer
    # bt = BERTTokenizer()
    
    # texts = [
    # "a photo of a cat",
    # "the quick brown fox jumps",
    # "an example sentence"
    # ]

    # output = bt(texts)
    # print(output.shape)


# ------------------------------------------------------
                ##### BERTEmbedder 
    
    # embedder = BERTEmbedder(n_embed=512,
    #                         n_layer=6).cuda()
    
    # texts = "a photo of a cat"
    
    
    # output = embedder(texts)
    # print(output.shape)

# -----------------------------------------------------

    # from torchvision.io import read_image
    # from torchvision.utils import save_image

    #         ### SpatialRescaler
    # # Downscaler: 2 stages of 0.5x downscaling (total 0.25x) + channel remapping
    # downscaler = SpatialRescaler(
    #     n_stages=2,
    #     multiplier=0.5,
    #     in_channels=3,
    #     out_channels=64,
    #     method='bilinear'
    # )

    # # Upscaler: 1 stage of 2x upscaling
    # upscaler = SpatialRescaler(
    #     n_stages=1,
    #     multiplier=2.0,
    #     method='bicubic'
    # )

    # print("Downscaler configured:", downscaler)
    # print("Upscaler configured:", upscaler)

    # # 2. Load sample image
    # input_img = read_image("clip/test.png")  # Shape: [3, H, W]
    # input_img = input_img.float() / 255.0  # Convert to [0,1] range
    # input_img = input_img.unsqueeze(0)    # Add batch dim: [1, 3, H, W]

    # # 3. Process with downscaler
    # downscaled = downscaler(input_img)
    # print("Downscaled output shape:", downscaled.shape)  # [1, 64, H//4, W//4]

    # # 4. Process with upscaler
    # upscaled = upscaler(input_img)
    # print("Upscaled output shape:", upscaled.shape)      # [1, 3, H*2, W*2]

    # # 5. Save results for visualization
    # save_image(input_img, 'original.jpg')
    # save_image(downscaled[:, :3, ...], 'downscaled.jpg')  # Show first 3 channels
    # save_image(upscaled, 'upscaled.jpg')

# ----------------------------------------------------------------------------------
                ### FrozenCLIPEmbedder

    # fe = FrozenCLIPEmbedder().cuda()
    # text = "This is a cat"
    # output = fe(text)
    # print(output.shape)

# -----------------------------------------------------------------------

    from torchvision.io import read_image
    from torchvision.transforms.functional import convert_image_dtype
            ### FrozenClipImageEmbedder 
    # 1. Initialize the embedder
    model_name = 'ViT-B/32'  # CLIP model variant
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = FrozenClipImageEmbedder(
        model=model_name,
        jit=False,  # Don't use JIT-compiled model
        device=device,
        antialias=True  # Use anti-aliasing in resize
    ).to(device)
    embedder.eval()  # Set to evaluation mode

    print(f"Initialized {model_name} embedder on {device}")

    # 2. Load and prepare sample image
    def load_image(path):
        img = read_image(path)  # [C, H, W] uint8
        img = convert_image_dtype(img, torch.float)  # [0, 1] range
        img = img * 2 - 1  # Convert to [-1, 1] range (required by embedder)
        return img.unsqueeze(0).to(device)  # Add batch dim

    input_img = load_image("clip_encoder/test.png")
    print("Input image shape:", input_img.shape)  # [1, 3, H, W]
    print("Input value range:", input_img.min().item(), "to", input_img.max().item())

    # 3. Get image embeddings
    with torch.no_grad():
        embeddings = embedder(input_img)

    print("\nOutput embeddings shape:", embeddings.shape)  # [1, 512] for ViT-B/32

