import torch

def convert_and_save_fp16(ckpt_path, save_path):
    # Load checkpoint to CPU
    checkpoint = torch.load(ckpt_path, map_location="cpu")['state_dict']

    # Remove 'model.' prefix if present
    if any(k.startswith('model.') for k in checkpoint.keys()):
        checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}

    # Convert all float32 tensors to float16
    checkpoint_fp16 = {k: v.half() if v.dtype == torch.float32 else v for k, v in checkpoint.items()}

    # Save the new checkpoint
    torch.save({'state_dict': checkpoint_fp16}, save_path)
    print(f"Saved float16 weights to: {save_path}")

if __name__ == "__main__":
    ckpt_path = "vqvae/vq-f4/model.ckpt"  # original checkpoint path
    save_path = "vqvae/vq-f4/model_fp16.ckpt"  # new fp16 checkpoint path
    convert_and_save_fp16(ckpt_path, save_path)