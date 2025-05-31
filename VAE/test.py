import torch 
from torch import nn 
from VAE.autoencoder import AutoEncoderKL
from Dataset.lsun import LSUNBase
from torch.utils.data import DataLoader
import yaml






# -----------------------------------------------------------------------------------------------------------



# path = "diffusion_pytorch_model.fp16.safetensors"


# from safetensors.torch import load_file

# loaded = load_file(path)
# # print(loaded)

# for key, tensor in loaded.items():
#     print("key: ", key)
#     print("tensor: ", tensor.shape)

#     exit()

# -----------------------------------------------------


ckpt_path = "lightning_logs/version_0/checkpoints/epoch=9-step=9610.ckpt"
config_path = "config/vae_config/kl-f4.yaml" 



import torch 
from torch import nn 
import numpy as np 
from PIL import Image 
import torchvision.transforms as T 
import matplotlib.pyplot as plt 



# 1. Load you model architecture 
def load_model_from_config(config_path):

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = AutoEncoderKL(ddconfig=config['model']['params']['ddconfig'],
                                embed_dim=config['model']['embed_dim'],
                                monitor="val/total_loss",
                                lossconfig=config['model']['params']['lossconfig']
                                # lossconfig=None,
                                )
    
    return model 




# 2. Load checkpoint with device handling
def load_checkpoint(model, ckpt_path, device="cuda"):
    # Load checkpoint to CPU first for safety
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint['state_dict']
    
    # Fix common key prefix issues
    if any(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    
    # Check for NaN/Inf in checkpoint
    checkpoint_issues = False
    for name, tensor in state_dict.items():
        if torch.isnan(tensor).any():
            print(f"❌ NaN found in checkpoint: {name}")
            checkpoint_issues = True
        if torch.isinf(tensor).any():
            print(f"❌ Inf found in checkpoint: {name}")
            checkpoint_issues = True
    
    # Load state dict
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"⚠️ Missing keys: {missing}")
    if unexpected:
        print(f"⚠️ Unexpected keys: {unexpected}")
    
    # Move model to target device
    model = model.to(device)
    model.eval()
    
    return model, checkpoint_issues

# 3. Verify model parameter health
def check_model_health(model):
    print("\n🔍 Running model health check...")
    issues_found = False
    
    # Check parameters
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"❌ NaN in parameter: {name}")
            issues_found = True
        if torch.isinf(param).any():
            print(f"❌ Inf in parameter: {name}")
            issues_found = True
    
    # Check buffers
    for name, buffer in model.named_buffers():
        if torch.isnan(buffer).any():
            print(f"❌ NaN in buffer: {name}")
            issues_found = True
        if torch.isinf(buffer).any():
            print(f"❌ Inf in buffer: {name}")
            issues_found = True
    
    if not issues_found:
        print("✅ All parameters and buffers are clean")
    
    return issues_found


# 4. Test forward pass with device awareness
def test_forward_pass(model, device="cuda", use_real_image=False):
    print("\n🚀 Testing forward pass...")
    
    # Create test input on the correct device
    if use_real_image:
        try:
            img = Image.open("VAE/test/test_image.jpg").convert("RGB")
            transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(256),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            x = transform(img).unsqueeze(0).to(device)
        except Exception as e:
            print(f"⚠️ Image loading failed: {str(e)}")
            print("Using random input instead")
            x = torch.randn(1, 3, 256, 256).to(device)
    else:
        x = torch.randn(1, 3, 256, 256).to(device)
    
    # Run forward pass
    try:
        with torch.no_grad():
            reconstructions, posterior = model(x)

        
        
        # Check outputs
        output_issues = False
        if torch.isnan(reconstructions).any():
            print("❌ NaN in reconstructions output")
            output_issues = True
        if torch.isinf(reconstructions).any():
            print("❌ Inf in reconstructions output")
            output_issues = True
        if hasattr(posterior, 'mean') and (torch.isnan(posterior.mean).any() or torch.isnan(posterior.logvar).any()):
            print("❌ NaN in latent distribution")
            output_issues = True
        
        if not output_issues:
            print("✅ Forward pass successful with clean outputs")
            print(f"Reconstruction shape: {reconstructions.shape}")
            print(f"Reconstruction range: [{reconstructions.min().item():.4f}, {reconstructions.max().item():.4f}]")
            
            # Return sample for visualization
            return reconstructions.squeeze(0).cpu().numpy()
        return None
    except Exception as e:
        print(f"❌ Forward pass failed: {str(e)}")
        return None

# 5. Visualize results
def visualize_results(original, reconstruction):
    
    
    # Process original
    original = original.permute(1, 2, 0).cpu().numpy()
    original = (original * 0.5 + 0.5).clip(0, 1)
    
    # Process reconstruction
    if reconstruction.dtype == np.float16:
        reconstruction = reconstruction.astype(np.float32)

    reconstruction = reconstruction.transpose(1, 2, 0)
    reconstruction = (reconstruction * 0.5 + 0.5).clip(0, 1)
    reconstruction = (reconstruction * 255).astype(np.uint8)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(original)
    ax[0].set_title("Original")
    ax[0].axis('off')
    
    ax[1].imshow(reconstruction)
    ax[1].set_title("Reconstruction")
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("reconstruction_test.png")
    print("💾 Saved visualization as 'reconstruction_test.png'")

# Main testing function
def test_checkpoint(config_path, ckpt_path):
    print("🧪 Starting checkpoint validation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load model architecture
    model = load_model_from_config(config_path)
    print("✅ Model architecture loaded")
    
    # 2. Load checkpoint
    model, ckpt_issues = load_checkpoint(model, ckpt_path, device)
    print(f"Checkpoint loaded with {'issues' if ckpt_issues else 'no issues'}")
    
    # 3. Parameter health check
    model_issues = check_model_health(model)
    
    # 4. Forward pass test
    print("\n🔧 Testing with random input...")
    test_result = test_forward_pass(model, device)
    
    print("\n🔧 Testing with real image...")
    img_result = test_forward_pass(model, device, use_real_image=True)
    
    # 5. Visualization
    if img_result is not None:
        try:
            img = Image.open("VAE/test/test_image.jpg").convert("RGB")
            transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(256),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            original_img = transform(img)
            visualize_results(original_img, img_result)
        except Exception as e:
            print(f"⚠️ Visualization failed: {str(e)}")
    
    # Summary
    print("\n📊 Final Validation Summary:")
    print(f"1. Checkpoint loaded: {'❌ Failed' if ckpt_issues else '✅ Passed'}")
    print(f"2. Parameter health: {'❌ Failed' if model_issues else '✅ Passed'}")
    print(f"3. Random input test: {'❌ Failed' if test_result is None else '✅ Passed'}")
    print(f"4. Real image test: {'❌ Failed' if img_result is None else '✅ Passed'}")
    
    if not ckpt_issues and not model_issues and test_result is not None and img_result is not None:
        print("\n🎉 Checkpoint PASSED all validation tests!")
    else:
        print("\n🔴 Checkpoint FAILED one or more tests")




        
if __name__ == "__main__":
    
    result = test_checkpoint(config_path=config_path, ckpt_path=ckpt_path)
    print(result)

    