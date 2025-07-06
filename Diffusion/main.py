from Diffusion.utils import instantiate_from_config
import yaml
from Diffusion.ddpm import LatentDiffusion
from Diffusion.data.dataset import DataModuleFromConfig
import torch, os

# Override checkpoint globally
def safe_checkpoint(function, *args, **kwargs):
    return function(*args, **kwargs)

torch.utils.checkpoint.checkpoint = safe_checkpoint

# Set environment variable as extra precaution
os.environ['TORCH_CHECKPOINT_DISABLE'] = '1'





if __name__ == "__main__":

    config = "Diffusion/config.yaml" 

        # Load the YAML file 
    with open(config, 'r') as file:
        config = yaml.safe_load(file)

    # Explicitly disable checkpointing in all model components
    def disable_checkpointing(config_part):
        print(f"what is the data to get config: {config_part}")
        if isinstance(config_part, dict):
            for key in list(config_part.keys()):
                if key in ["use_checkpoint", "use_gradient_checkpointing"]:
                    config_part[key] = False
                else:
                    disable_checkpointing(config_part[key])
        elif isinstance(config_part, list):
            for item in config_part:
                disable_checkpointing(item)
    
    disable_checkpointing(config)
  
    from Diffusion.data.coco import CocoDataset



   
    from torch.utils.data import DataLoader
    from torchvision import transforms
    

    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor()
    ])


    # initial dataset 
    train_dataset = CocoDataset(
        root_dir = "coco_data/train2017",
        annotation_file="coco_data/captions_train2017.json",
        transform=transform
    )
    val_dataset = CocoDataset(
        root_dir = "coco_data/val2017",
        annotation_file="coco_data/captions_val2017.json",
        transform=transform
    )

    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        txt = [item[1] for item in batch]

        return {
            "image": images,
            "txt": txt
        }


    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4

    )

    def safe_checkpoint(function, *args, **kwargs):
        return function(*args, **kwargs)

    torch.utils.checkpoint.checkpoint = safe_checkpoint



    import pytorch_lightning as pl

    model = LatentDiffusion(
            first_stage_config=config["model"]["params"]["first_stage_config"],
            cond_stage_config=config["model"]["params"]["cond_stage_config"],
            unet_config = config["model"]["params"]["unet_config"],
            **{k: v for k, v in config["model"]["params"].items() if k not in ["unet_config", "first_stage_config", "cond_stage_config"]}
        ).to("cuda:0")
    
 




    trainer = pl.Trainer(
        max_epochs=10,
        devices=1,
        accelerator="gpu",
        precision="16-mixed"
    )


    trainer.fit(model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                )




    