from Diffusion.utils import instantiate_from_config
import yaml
from Diffusion.ddpm import LatentDiffusion
from Diffusion.data.dataset import DataModuleFromConfig

if __name__ == "__main__":

    config = "Diffusion/config.yaml" 

        # Load the YAML file 
    with open(config, 'r') as file:
        config = yaml.safe_load(file)

    # print(f"config: >>>> {config} ")

    # Build components from configs 
    # first_stage_model = instantiate_from_config(config["model"]["params"]["first_stage_config"])
    # print(f"first stage config: >>>> {first_stage_model}")

    # cond_stage_model = instantiate_from_config(config["model"]["params"]["cond_stage_config"])
    # print(f"cond stage config: >>>>> {cond_stage_model}")

    # unet_model = instantiate_from_config(config["model"]["params"]["unet_config"])
    # print(f"unet model >>>>> {unet_model}")


    # Assemble LatentDiffusion
    model = LatentDiffusion(
        first_stage_config=config["model"]["params"]["first_stage_config"],
        cond_stage_config=config["model"]["params"]["cond_stage_config"],
        unet_config = config["model"]["params"]["unet_config"],
        **{k: v for k, v in config["model"]["params"].items() if k not in ["unet_config", "first_stage_config", "cond_stage_config"]}
    )

    # print(f"model: >>>> {model}")

    from Diffusion.data.imagenet import ImageNetTrain, ImageNetValidation, ImageNetSRTrain, ImageNetSRValidation

    # Example configuration
    config = {
        "size": 256,          # Resize images to 256x256
        "random_crop": True,  # Use random cropping for training
    }

    train_dataset = ImageNetSRTrain(
        size=256
    )

    val_dataset = ImageNetSRValidation(
        size=256
    )

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        num_workers=8,
    )

    import pytorch_lightning as pl

    class LatentDiffusionTrainer(pl.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def training_step(self, batch, batch_idx):
            loss, loss_dict = self.model.shared_step(batch)
            self.log_dict(loss_dict, prog_bar=True, logger=True)
            return loss

        def validation_step(self, batch, batch_idx):
            _, loss_dict = self.model.shared_step(batch)
            self.log_dict(loss_dict, prog_bar=False, logger=True)

        def configure_optimizers(self):
            return model.configure_optimizers()  # Uses AdamW with LR scheduler

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=100,                       
        strategy="ddp",                   # Distributed training
        precision=16,                    # Mixed precision
        log_every_n_steps=100,
    )

    # Start training
    trainer.fit(
        LatentDiffusionTrainer(model),
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )




    