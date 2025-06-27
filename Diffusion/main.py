from Diffusion.utils import instantiate_from_config
import yaml
from Diffusion.ddpm import LatentDiffusion
from Diffusion.data.dataset import DataModuleFromConfig
import torch 



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


    

    # print(f"model: >>>> {model}")

    # from Diffusion.data.imagenet import ImageNetTrain, ImageNetValidation, ImageNetSRTrain, ImageNetSRValidation
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



    import pytorch_lightning as pl

    model = LatentDiffusion(
            first_stage_config=config["model"]["params"]["first_stage_config"],
            cond_stage_config=config["model"]["params"]["cond_stage_config"],
            unet_config = config["model"]["params"]["unet_config"],
            # conditioning_key=config["model"]["params"]["conditioning_key"],
            **{k: v for k, v in config["model"]["params"].items() if k not in ["unet_config", "first_stage_config", "cond_stage_config"]}
        ).to("cuda:0")
    
    
    # for batch in train_loader:
    #     print(f"check the batch size: >>> {batch['image'].shape}")
    #     print(f"check the caption of image: >>>> {batch['txt']}")
    #     # break

    #     output = model(batch["image"],
    #                    batch["txt"])
        
    #     break
    


    trainer = pl.Trainer(
        max_epochs=10,
        devices=1,
        accelerator="gpu",
        precision="16-mixed"
    )


    trainer.fit(model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)




    