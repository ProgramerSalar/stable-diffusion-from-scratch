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

    data_module_from_config = DataModuleFromConfig(batch_size=)




    