import torch 
from torch import nn 
from VAE.autoencoder import AutoEncoderKL
from Dataset.lsun import LSUNBase
from torch.utils.data import DataLoader
import yaml








config_path = "config/vae_config/kl-f4.yaml" 



import torch 
from torch import nn 
import numpy as np 
from PIL import Image 
import torchvision.transforms as T 
import matplotlib.pyplot as plt 
from safetensors.torch import load_file


# 1. Load you model architecture 
def load_model_from_config(config_path):

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = AutoEncoderKL(ddconfig=config['model']['params']['ddconfig'],
                                embed_dim=config['model']['embed_dim'],
                                monitor="val/total_loss",
                                lossconfig=config['model']['params']['lossconfig']
                                # lossconfig=None,
                                ).half()
    
    return model 



# 2. Load checkpoint with device handling 
def load_checkpoint(model, ckpt_path, device="cuda"):

    # Load checkpoint to cpu first for safety 
    checkpoint = torch.load(ckpt_path, map_location="cpu")['state_dict'] # this is load to ckpt path
    # checkpoint = load_file(filename=ckpt_path, device="cpu")  # this is load to safetensors path
    # for name, tensor in checkpoint:
        # print(f"model name:  {name} and shape : {tensor.shape}")

    


    # Fix common key prefix issues 
    if any(k.startswith('model.') for k in checkpoint.keys()):
        checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}


    # check for NaN/Inf in checkpoint 
    checkpoint_issues = False 
    for name, tensor in checkpoint.items():
        if torch.isnan(tensor).any():
            print(f"❌ NaN found in checkpoint: {name}")
            checkpoint_issues = True

        if torch.isinf(tensor).any():
            print(f"❌ Inf found in checkpoint: {name}")
            checkpoint_issues = True

    # Load state dict 
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    # print("what is the output comming from this missing [var]: ", missing)
    # print("what is the output comming from this unexpected [var]: ", unexpected)

    # checking = model.load_state_dict(checkpoint, strict=False)
    # print("checking: ", checking)

    if missing:
        print(f"⚠️ Missing keys: {missing}")

    if unexpected:
        print(f"⚠️ Unexpected keys: {unexpected}")

    
    # Move model to target device 
    model = model.to(device)
    model.eval()

    return model, checkpoint_issues

# 3. verify model parameter health 
def check_model_health(model):

    print("\n 🔍 Running model health check...")
    issues_found = False 

    # check parameters 
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"❌ NaN is parameter: {name}")
            issues_found = True

        if torch.isinf(param).any():
            print(f"❌ Inf in parameter: {name}")
            issues_found = True

    # check buffers 
    for name, buffer in model.named_buffers():
        if torch.isnan(buffer).any():
            print(f"❌ NaN in buffer: {name}")
            issues_found = True

        if torch.isinf(buffer).any():
            print(f"❌ Inf in buffer: {name}")
            issues_found = True

        
    if not issues_found:
        print(f"✅ All parameters and buffers are clean")

    return issues_found


# 4. Test forward pass with device awarness 
def test_forward_pass(model, device="cuda", use_real_image=False):
    print("\n 🚀 Testing forward pass...")

    # create test input on the correct device 
    if use_real_image:
        try:
            img = Image.open('VAE/test/Image/test_image.jpg')
            transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(256),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            x = transform(img).unsqueeze(0).to(device).half()
            # print("real Image: ", x)

        except Exception as e:
            print(f"⚠️ Image loading failed: {str(e)}")
            print("Using random input instead.")
            x = torch.randn(1, 3, 256, 256).to(device)

    else:
        x = torch.randn(1, 3, 256, 256).to(device)


    # Run forward pass 
    try:
        with torch.no_grad():
            reconstructions, posterior = model(x)

            # check outputs 
            output_issues = False 
            if torch.isnan(reconstructions).any():
                print("❌ NaN in reconstructions output")
                output_issues = True

            if torch.isinf(reconstructions).any():
                print("❌ Inf in reconstructions output")
                output_issues = True

            if hasattr(posterior, "mean") and (torch.isnan(posterior.mean).any() or torch.isnan(posterior.logvar).any()):
                print("❌ NaN in latent distribution")
                output_issues = True

            if not output_issues:
                print("✅ Forward pass successful with clean outputs")
                print(f"Reconstruction shape: {reconstructions.shape}")
                print(f"Reconstruction range: [{reconstructions.min().item():.4f}, {reconstructions.max().item():4f}]")

                # Return sample for visualization 
                return reconstructions.squeeze(0).cpu().numpy()
            
            return None 
        


    except Exception as e:
        print(f"❌ Forward pass failed: {str(e)}")
        return None
    


# 5. visualize results 
def visualize_results(original, reconstruction):

    

    # process original 
    original = original[0]
    original = original.permute(1, 2, 0).cpu().numpy()
    original = (original * 0.5 + 0.5).clip(0, 1)
    print("original Image: ", original.dtype)

    # process reconstruction
    if reconstruction.dtype == np.float16:
        reconstruction = reconstruction.astype(np.float32)
        print("Reconstruction Image: ", reconstruction.dtype)

    # print("what is the shape of reconstucted image: ", reconstruction.shape)
    reconstruction = reconstruction.transpose(1, 2, 0)
    reconstruction = (reconstruction * 0.5 + 0.5).clip(0, 1)
    # reconstruction = (reconstruction * 255)
    # print("Reconstructed Image: ", reconstruction.shape)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(original)
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(reconstruction)
    ax[1].set_title("Reconstructed")
    ax[1].axis("off")

    # save to image 
    temp_path = "VAE/test/reconstruction_image.png"
    plt.savefig(temp_path, bbox_inches="tight", dpi=80)
    plt.close(fig)




    



        
if __name__ == "__main__":

    # ckpt_path = "lightning_logs/version_0/checkpoints/epoch=9-step=9610.ckpt"     # This is my weight which i was train epochs==10
    ckpt_path = "VAE/models/kl-f4/model.ckpt"       # This is stable-diffusion weight 
    

    
    model = load_model_from_config(config_path=config_path)
    checkpoint_tensor = load_checkpoint(model=model, ckpt_path=ckpt_path)
    THE_check_model_health = check_model_health(model=model)
    THE_test_forward_path = test_forward_pass(model=model, use_real_image=True)

    img = Image.open('VAE/test/Image/test_image.jpg')
    transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(256),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    x = transform(img).unsqueeze(0).cuda()
    # print("real Image: ", x)

    THE_result_visualize = visualize_results(original=x, reconstruction=THE_test_forward_path)
    print(THE_result_visualize)

    


    