import datetime
import sys, os, argparse
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import Callback
from torch import nn 
from omegaconf import OmegaConf
import pytorch_lightning as pl
import numpy as np 
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_info
import torchvision
from PIL import Image
import torch 
import time, glob

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer

def get_parser(**parser_kwargs):

    def str2bool(v):
        if isinstance(v, bool):
            return v 
        

        if v.lower() in ("yes", "true", "t", "y", "i"):
            return True
        
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
        

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir"
    )        
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir"
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="/Diffusion/config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
                "Parameters can be overwritten or added with command-line options of the form `--key value`. ",
        default=list()
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train"
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test"
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging"
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything"
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name"
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit"
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate"
    )

    parser.add_argument("--accelerator", type=str, default=None, help="Supports passing different accelerator types")
    parser.add_argument("--devices", type=int, default=None, help="Number of devices to use")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes to use")
    parser.add_argument("--strategy", type=str, default=None, help="Training strategy (ddp, ddp_spawn, etc)")
    parser.add_argument("--precision", type=int, default=32, help="Precision (16 or 32)")
    parser.add_argument("--max_epochs", type=int, default=1000, help="Maximum number of epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of steps")
    parser.add_argument("--gradient_clip_val", type=float, default=0, help="Gradient clipping value")


    

    return parser



def nondefault_trainer_args(opt):
    default_args = {
        "accelerator": None,
        "devices": None,
        "num_nodes": 1,
        "strategy": None,
        "precision": 32,
        "max_epochs": 1000,
        "max_steps": -1,
        "gradient_clip_val": 0,
        # Add other defaults here...
    }
    return sorted(k for k in default_args if getattr(opt, k) != default_args[k])


class SetupCallback(Callback):

    def __init__(self,
                 resume,
                 now,
                 logdir,
                 ckptdir,
                 cfgdir,
                 config,
                 lightninig_config):
        

        super().__init__()
        self.resume = resume        # whether we are resuming training
        self.now = now              # Timestep for unique naming 
        self.logdir = logdir        # Root log directory
        self.ckptdir = ckptdir      # checkpoint directory
        self.cfgdir = cfgdir        # config directory
        self.config = config        # Main configuration object
        self.lightning_config = lightninig_config   # pytorch lightning config

    def on_keyboard_interrupt(self, trainer, pl_module):

        if trainer.global_rank == 0:            # only execute on main procses
            print("Summong checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)      # save the checkpoint in emergency condition



    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:    # only main process handles setup
            # create directory if missing 
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)


            # Special handling for step-based checkpointing 
            if "callback" in self.lightning_config:
                if "matrics_over_trainsteps_checkpoint" in self.lightning_config["callbacks"]:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)

            # save project config 
            print("Project config")
            print(OmegaConf.to_yaml(self.config))

            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, f"{self.now}-project.yaml"))
            
            # save lightning config 
            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, f"{self.now}-lightning.yaml"))
            


        else:

            # non-main process clean up potential directory conflicts 
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)

                os.makedirs(os.path.split(dst)[0], exist_ok=True)

                try:
                    os.rename(self.logdir, dst)

                except FileNotFoundError:
                    pass 


        
class ImageLogger(Callback):

    def __init__(self,
                 batch_frequency,
                 max_images,
                 clamp=True,
                 increase_log_steps=True,
                 rescale=True,
                 disabled=False,
                 log_on_batch_idx=False,
                 log_first_step=False,
                 log_images_kwargs=None):
        
        super().__init__()
        self.rescale = rescale      # whether to rescale images from [-1, 1] to [0, 1]
        self.batch_freq = batch_frequency   # Log every N batches 
        self.max_images = max_images        # max images to log per batch 

        # Logger-specific image logging functions 
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube
        }

        # Exponential logging schedule (2, 4, 8, ... until batch_freq)
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]

        if not increase_log_steps:
            self.log_steps = [self.batch_freq]  # fixed interval logging 

        self.clamp = clamp      # clamp image values to [-1, 1]
        self.disabled = disabled    # Enable/disable callback

        self.log_on_batch_idx = log_on_batch_idx    # use batch idx instead of global step 
        self.log_images_kwargs = log_images_kwargs or {}    # Extra args for log_images 
        self.log_first_step = log_first_step    # Log at step 0 


    # Logger-specific image Handler (TestTube/TensorBoard)
    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):

        for k in images:    # Process each image type (inputs, reconstructions, samples)
            grid = torchvision.utils.make_grid(images[k])   # create image grid 
            grid = (grid + 1.0) / 2.0   # convert from [-1, 1] to [0, 1]
            tag = f"{split}/{k}"    # e.g.. "train/reconstructions"

            pl_module.logger.experiment.add_image(
                tag,
                grid,
                global_step=pl_module.global_step       # Log to TensorBoard
            )


    # Local image saver 
    @rank_zero_only
    def log_local(self, 
                  save_dir,
                  split,
                  images,
                  global_step,
                  current_epoch,
                  batch_idx):
        

        root = os.path.join(save_dir, "images", split)  # e.g... logs/exp1/images/train

        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)   # 4 images per row 
            if self.rescale:
                grid = (grid + 1.0) / 2.0   # Rescale if enabled 

            # convert tensor to PIL image 
            grid = grid.transpose(0, 1).transpose(1, 2).square(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)

            # create filename with metadata 
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx
            )

            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)    # save as PNG



    # Main Image logging logic 
    def log_img(self, 
                pl_module,
                batch,
                batch_idx,
                split="train"):
        
        # Decide whether to use batch idx or global step for logging 
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step

        # check if we should log at this step 
        if (self.check_frequency(check_idx) and     # Logging schedule check
            hasattr(pl_module, "log_images") and    # Model must implement
            callable(pl_module.log_images) and  # Must be callable 
            self.max_images > 0):   # Must log at least 1 image

            logger = type(pl_module.logger)     # Get logger type 

            # Switch to eval model for image generation 
            is_train = pl_module.training 
            if is_train:
                pl_module.eval()

            
            # Generate images without gradients 
            with torch.no_grad():
                images = pl_module.log_images(
                    batch,
                    split=split,
                    **self.log_images_kwargs
                )


            # Process images 
            for k in images:
                N = min(images[k].shape[0], self.max_images)    # limit images 
                images[k] = images[k][:N]

                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()    # To CPU
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)     # Clamp values 



            # Save locally and log to logger 
            self.log_local(pl_module.logger.save_dir, 
                           split,
                           images,
                           pl_module.global_step,
                           pl_module.current_epoch,
                           batch_idx)
            

            # logger-specific logging 
            logger_log_images = self.logger_log_images.get(logger, lambda *a, **k: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)


            # Restore training mode if needed 
            if is_train:
                pl_module.train()


        # Training hook 
    def on_train_batch_end(self,
                           trainer,
                           pl_module,
                           outputs,
                           batch,
                           batch_idx,
                           dataloader_idx):
        
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")



    def on_validation_batch_end(self,
                                trainer,
                                pl_module,
                                outputs,
                                batch,
                                batch_idx,
                                dataloader_idx):
        
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")

        if hasattr(pl_module, "clibrate_grad_norm"):
            if (pl_module.callibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx)



    def check_frequency(self,
                        check_idx):
        
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
            check_idx > 0 or self.log_first_step
        ):
            
            try:
                self.log_steps.pop(0)

            except IndexError as e:
                print(e)
                pass 

            return True
        
        return False
    


class CUDACallback(Callback):

    def on_train_epoch_start(self,
                             trainer,
                             pl_module):
        
        # Rest the memory use counter 
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()


    def on_train_epoch_end(self, 
                           trainer,
                           pl_module,
                           outputs):
        
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20 
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time: .2f} seconds")
            rank_zero_info(f"Average Peak memory: {max_memory: .2f} MIB")

        except AttributeError:
            pass 






        





if __name__ == "__main__":

    # generate a timesteps for unique naming 
    now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    sys.path.append(os.getcwd())

    
    parser = get_parser()

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder."
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError(f"Cannot find {opt.resume}")
        
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume 

        else:
            assert os.path.isdir(opt.resume), opt.resume 
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoint", "last.ckpt")


        opt.resume_from_checkpoint = ckpt
        
        base_configs = sorted(glob.glob(os.path.join(logdir, "Diffusion/config.yaml")))
        opt.base = base_configs + opt.base 
        _tmp = logdir.split("/")
        nowname = _tmp[-1]

    else:
        if opt.name:
            name = "_" + opt.name 

        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
            
        else:
            name = ""


        nowname = now + name + opt.postfix 
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)


    try:
        print(f"what is the output i get in opt.base : >>>> {opt}")
        
        # init and save configs 
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        print("configs: ", configs)

    except Exception:
        pass


    


    