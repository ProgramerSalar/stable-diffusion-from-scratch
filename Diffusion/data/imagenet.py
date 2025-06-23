import torch, yaml, os, tarfile, glob, shutil
from torch.utils.data import Dataset, Subset, DataLoader
from omegaconf import OmegaConf
import albumentations
import cv2
from functools import partial
import PIL
import torchvision.transforms.functional as F
# from image_degradation import degradation_fn_bsr
from image_degradation.bsrgan import degradation_bsrgan_variant as degradation_fn_bsr
from image_degradation.bsrgan_light import degradation_bsrgan_variant as degradation_fn_bsr_light
from PIL import Image
import pickle

from Diffusion.data.utils import (
    str_to_indices, 
    give_synsets_from_indices, 
    download, 
    retrieve, 
    is_prepared,
    mark_prepared,
    
)


import numpy as np
from Diffusion.data.base import ImagePaths
from tqdm import tqdm


def synset2idx(path_to_yaml="data/index_synset.yaml"):
    with open(path_to_yaml) as f:
        di2s = yaml.load(f)

    return dict((v, k) for k, v in di2s.items())




class ImageNetBase(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)
        self.keep_orig_class_label = self.config.get("keep_orig_class_label", False)
        self.process_images = True  # if False we skip loading & processing images and self.data contains filepaths
        self._prepare()
        self._prepare_synset_to_human()
        self._prepare_idx_to_synset()
        self._prepare_human_to_integer_label()
        self._load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def _prepare(self):
        raise NotImplementedError()

    def _filter_relpaths(self, relpaths):
        ignore = set([
            "n06596364_9591.JPEG",
        ])
        relpaths = [rpath for rpath in relpaths if not rpath.split("/")[-1] in ignore]
        if "sub_indices" in self.config:
            indices = str_to_indices(self.config["sub_indices"])
            synsets = give_synsets_from_indices(indices, path_to_yaml=self.idx2syn)  # returns a list of strings
            self.synset2idx = synset2idx(path_to_yaml=self.idx2syn)
            files = []
            for rpath in relpaths:
                syn = rpath.split("/")[0]
                if syn in synsets:
                    files.append(rpath)
            return files
        else:
            return relpaths

    def _prepare_synset_to_human(self):
        SIZE = 2655750
        URL = "https://heibox.uni-heidelberg.de/f/9f28e956cd304264bb82/?dl=1"
        self.human_dict = os.path.join(self.root, "synset_human.txt")
        if (not os.path.exists(self.human_dict) or
                not os.path.getsize(self.human_dict)==SIZE):
            download(URL, self.human_dict)

    def _prepare_idx_to_synset(self):
        URL = "https://heibox.uni-heidelberg.de/f/d835d5b6ceda4d3aa910/?dl=1"
        self.idx2syn = os.path.join(self.root, "index_synset.yaml")
        if (not os.path.exists(self.idx2syn)):
            download(URL, self.idx2syn)

    def _prepare_human_to_integer_label(self):
        URL = "https://heibox.uni-heidelberg.de/f/2362b797d5be43b883f6/?dl=1"
        self.human2integer = os.path.join(self.root, "imagenet1000_clsidx_to_labels.txt")
        if (not os.path.exists(self.human2integer)):
            download(URL, self.human2integer)
        with open(self.human2integer, "r") as f:
            lines = f.read().splitlines()
            print(f"what is the length of split lines: {len(lines)}")
            assert len(lines) == 1000
            self.human2integer_dict = dict()
            for line in lines:
                value, key = line.split(":")
                self.human2integer_dict[key] = int(value)

    def _load(self):
        with open(self.txt_filelist, "r") as f:
            self.relpaths = f.read().splitlines()
            l1 = len(self.relpaths)
            self.relpaths = self._filter_relpaths(self.relpaths)
            print("Removed {} files from filelist during filtering.".format(l1 - len(self.relpaths)))

        self.synsets = [p.split("/")[0] for p in self.relpaths]
        self.abspaths = [os.path.join(self.datadir, p) for p in self.relpaths]

        unique_synsets = np.unique(self.synsets)
        class_dict = dict((synset, i) for i, synset in enumerate(unique_synsets))
        if not self.keep_orig_class_label:
            self.class_labels = [class_dict[s] for s in self.synsets]
        else:
            self.class_labels = [self.synset2idx[s] for s in self.synsets]

        with open(self.human_dict, "r") as f:
            human_dict = f.read().splitlines()
            human_dict = dict(line.split(maxsplit=1) for line in human_dict)

        self.human_labels = [human_dict[s] for s in self.synsets]

        labels = {
            "relpath": np.array(self.relpaths),
            "synsets": np.array(self.synsets),
            "class_label": np.array(self.class_labels),
            "human_label": np.array(self.human_labels),
        }

        if self.process_images:
            self.size = retrieve(self.config, "size", default=256)
            self.data = ImagePaths(self.abspaths,
                                   labels=labels,
                                   size=self.size,
                                   random_crop=self.random_crop,
                                   )
        else:
            self.data = self.abspaths

        

class ImageNetTrain(ImageNetBase):
    NAME = "ILSVRC2012_train"
    URL = "http://www.image-net.org/challenges/LSVRC/2012/"
    AT_HASH = "a306397ccf9c2ead27155983c254227c0fd938e2"
    FILES = [
        "ILSVRC2012_img_train.tar",
    ]
    SIZES = [
        147897477120,
    ]

    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.process_images = process_images
        self.data_root = data_root
        super().__init__(**kwargs)

    def _prepare(self):
       
        # Determine root directory (custom or default cache location)
        if self.data_root:
            self.root = os.path.join(self.data_root, self.NAME)
            print(f"data root: {self.root}")

        else:
            print("directory not found")

        # set paths for data and file list 
        self.datadir = os.path.join(self.root, "data")
        print(f"datadir are found: {self.datadir} ")

        self.txt_filelist = os.path.join(self.root, "filelist.txt")
        print(f"is file list are found: {self.txt_filelist}")

        self.expected_length = 1281167
        self.random_crop = retrieve(self.config,
                                    "ImageNetTrain/random_crop",
                                    default=True)
        

        # if not is_prepared(self.root):

        #     print("Root file does not exists.")
        #     raise FileNotFoundError

        
        print("else condition are working...")
        filelist = glob.glob(os.path.join(self.datadir, "**", "*.JPEG"))
        # print(f"Image are found: {filelist}")
        filelist = [os.path.relpath(p, start=self.datadir) for p in filelist]
        # print(f"Image are found: {filelist}")
        filelist = sorted(filelist)
        filelist = "\n".join(filelist)+"\n"
        # print(f"Image are found: {filelist}")

        # with open(self.txt_filelist, 'r') as f:
        #     f.read(filelist)

        # print(f"Image are found: {filelist}")

        # Mark preparation complete
        mark_prepared(self.root)
        
        
        
# class ImageNetTrain(ImageNetBase):
#     NAME = "ILSVRC2012_train"
#     URL = "http://www.image-net.org/challenges/LSVRC/2012/"
#     AT_HASH = "a306397ccf9c2ead27155983c254227c0fd938e2"
#     FILES = [
#         "ILSVRC2012_img_train.tar",
#     ]
#     SIZES = [
#         147897477120,
#     ]

#     def __init__(self, process_images=True, data_root=None, **kwargs):
#         self.process_images = process_images
#         self.data_root = data_root
#         super().__init__(**kwargs)

#     def _prepare(self):
#         if self.data_root:
#             self.root = os.path.join(self.data_root, self.NAME)
#         else:
#             cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
#             self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)

#         self.datadir = os.path.join(self.root, "data")
#         self.txt_filelist = os.path.join(self.root, "filelist.txt")
#         self.expected_length = 1281167
#         self.random_crop = retrieve(self.config, "ImageNetTrain/random_crop",
#                                     default=True)
#         if not is_prepared(self.root):
#             # prep
#             print("Preparing dataset {} in {}".format(self.NAME, self.root))

#             datadir = self.datadir
#             if not os.path.exists(datadir):
#                 path = os.path.join(self.root, self.FILES[0])
#                 if not os.path.exists(path) or not os.path.getsize(path)==self.SIZES[0]:
#                     import academictorrents as at
#                     atpath = at.get(self.AT_HASH, datastore=self.root)
#                     assert atpath == path

#                 print("Extracting {} to {}".format(path, datadir))
#                 os.makedirs(datadir, exist_ok=True)
#                 with tarfile.open(path, "r:") as tar:
#                     tar.extractall(path=datadir)

#                 print("Extracting sub-tars.")
#                 subpaths = sorted(glob.glob(os.path.join(datadir, "*.tar")))
#                 for subpath in tqdm(subpaths):
#                     subdir = subpath[:-len(".tar")]
#                     os.makedirs(subdir, exist_ok=True)
#                     with tarfile.open(subpath, "r:") as tar:
#                         tar.extractall(path=subdir)

#             filelist = glob.glob(os.path.join(datadir, "**", "*.JPEG"))
#             filelist = [os.path.relpath(p, start=datadir) for p in filelist]
#             filelist = sorted(filelist)
#             filelist = "\n".join(filelist)+"\n"
#             with open(self.txt_filelist, "w") as f:
#                 f.write(filelist)

#             mark_prepared(self.root)


    



class ImageNetValidation(ImageNetBase):
    NAME = "ILSVRC2012_validation"
    URL = "http://www.image-net.org/challenges/LSVRC/2012/"
    AT_HASH = "5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5"
    VS_URL = "https://heibox.uni-heidelberg.de/f/3e0f6e9c624e45f2bd73/?dl=1"
    FILES = [
        "ILSVRC2012_img_val.tar",
        "validation_synset.txt",
    ]
    SIZES = [
        6744924160,
        1950000,
    ]

    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.data_root = data_root
        self.process_images = process_images
        super().__init__(**kwargs)

    def _prepare(self):
        if self.data_root:
            self.root = os.path.join(self.data_root, self.NAME)
        else:
            cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
            self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)
        self.datadir = os.path.join(self.root, "data")
        self.txt_filelist = os.path.join(self.root, "filelist.txt")
        self.expected_length = 50000
        self.random_crop = retrieve(self.config, "ImageNetValidation/random_crop",
                                    default=False)
        if not is_prepared(self.root):
            # prep
            print("Preparing dataset {} in {}".format(self.NAME, self.root))

            datadir = self.datadir
            if not os.path.exists(datadir):
                path = os.path.join(self.root, self.FILES[0])
                if not os.path.exists(path) or not os.path.getsize(path)==self.SIZES[0]:
                    import academictorrents as at
                    atpath = at.get(self.AT_HASH, datastore=self.root)
                    print(f"what is the path of atpath: >>> {atpath}")
                    print(f"what is the path of path: >>>> {path}")
                    assert atpath == path

                print("Extracting {} to {}".format(path, datadir))
                os.makedirs(datadir, exist_ok=True)
                with tarfile.open(path, "r:") as tar:
                    tar.extractall(path=datadir)

                vspath = os.path.join(self.root, self.FILES[1])
                if not os.path.exists(vspath) or not os.path.getsize(vspath)==self.SIZES[1]:
                    download(self.VS_URL, vspath)

                with open(vspath, "r") as f:
                    synset_dict = f.read().splitlines()
                    synset_dict = dict(line.split() for line in synset_dict)

                print("Reorganizing into synset folders")
                synsets = np.unique(list(synset_dict.values()))
                for s in synsets:
                    os.makedirs(os.path.join(datadir, s), exist_ok=True)
                for k, v in synset_dict.items():
                    src = os.path.join(datadir, k)
                    dst = os.path.join(datadir, v)
                    shutil.move(src, dst)

            filelist = glob.glob(os.path.join(datadir, "**", "*.JPEG"))
            filelist = [os.path.relpath(p, start=datadir) for p in filelist]
            filelist = sorted(filelist)
            filelist = "\n".join(filelist)+"\n"
            with open(self.txt_filelist, "w") as f:
                f.write(filelist)

            mark_prepared(self.root)



class ImageNetSR(Dataset):
    def __init__(self, size=None,
                 degradation=None, downscale_f=4, min_crop_f=0.5, max_crop_f=1.,
                 random_crop=True):
        """
        Imagenet Superresolution Dataloader
        Performs following ops in order:
        1.  crops a crop of size s from image either as random or center crop
        2.  resizes crop to size with cv2.area_interpolation
        3.  degrades resized crop with degradation_fn

        :param size: resizing to size after cropping
        :param degradation: degradation_fn, e.g. cv_bicubic or bsrgan_light
        :param downscale_f: Low Resolution Downsample factor
        :param min_crop_f: determines crop size s,
          where s = c * min_img_side_len with c sampled from interval (min_crop_f, max_crop_f)
        :param max_crop_f: ""
        :param data_root:
        :param random_crop:
        """
        self.base = self.get_base()
        assert size
        assert (size / downscale_f).is_integer()
        self.size = size
        self.LR_size = int(size / downscale_f)
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        assert(max_crop_f <= 1.)
        self.center_crop = not random_crop

        self.image_rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)

        self.pil_interpolation = False # gets reset later if incase interp_op is from pillow

        if degradation == "bsrgan":
            self.degradation_process = partial(degradation_fn_bsr, sf=downscale_f)

        elif degradation == "bsrgan_light":
            self.degradation_process = partial(degradation_fn_bsr_light, sf=downscale_f)

        else:
            print("This else block are worked..")

            # set default degradation if None 
            if degradation is None:
                degradation = "cv_bicubic"

            interpolation_fn = {
            "cv_nearest": cv2.INTER_NEAREST,
            "cv_bilinear": cv2.INTER_LINEAR,
            "cv_bicubic": cv2.INTER_CUBIC,
            "cv_area": cv2.INTER_AREA,
            "cv_lanczos": cv2.INTER_LANCZOS4,
            "pil_nearest": PIL.Image.NEAREST,
            "pil_bilinear": PIL.Image.BILINEAR,
            "pil_bicubic": PIL.Image.BICUBIC,
            "pil_box": PIL.Image.BOX,
            "pil_hamming": PIL.Image.HAMMING,
            "pil_lanczos": PIL.Image.LANCZOS,
            }[degradation]

            self.pil_interpolation = degradation.startswith("pil_")

            if self.pil_interpolation:
                self.degradation_process = partial(F.resize, size=self.LR_size, interpolation=interpolation_fn)

            else:
                self.degradation_process = albumentations.SmallestMaxSize(max_size=self.LR_size,
                                                                          interpolation=interpolation_fn)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        example = self.base[i]
        image = Image.open(example["file_path_"])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = np.array(image).astype(np.uint8)

        min_side_len = min(image.shape[:2])
        crop_side_len = min_side_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
        crop_side_len = int(crop_side_len)

        if self.center_crop:
            self.cropper = albumentations.CenterCrop(height=crop_side_len, width=crop_side_len)

        else:
            self.cropper = albumentations.RandomCrop(height=crop_side_len, width=crop_side_len)

        image = self.cropper(image=image)["image"]
        image = self.image_rescaler(image=image)["image"]

        if self.pil_interpolation:
            image_pil = PIL.Image.fromarray(image)
            LR_image = self.degradation_process(image_pil)
            LR_image = np.array(LR_image).astype(np.uint8)

        else:
            LR_image = self.degradation_process(image=image)["image"]

        example["image"] = (image/127.5 - 1.0).astype(np.float16)
        example["LR_image"] = (LR_image/127.5 - 1.0).astype(np.float16)

        return example 
    






class ImageNetSRTrain(ImageNetSR):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        with open("data/imagenet_train_hr_indices.p", "rb") as f:
            indices = pickle.load(f)

        dset = ImageNetTrain(data_root="data", process_images=False,)
        return Subset(dataset=dset, 
                      indices=indices)
    





class ImageNetSRValidation(ImageNetSR):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        with open("data/imagenet_val_hr_indices.p", "rb") as f:
            indices = pickle.load(f)

        dset = ImageNetValidation(data_root="data", process_images=False,)

        return Subset(dset, indices)
    




if __name__ == "__main__":
    # ImageNetTrain(data_root="data/")
    # ImageNetValidation(data_root="data/")

    output = ImageNetSRTrain(size=256)
    
    dataloader = DataLoader(dataset=output, batch_size=32, shuffle=True)

    for batch in dataloader:
        hr_images = batch["image"] 
        # print(hr_images.shape)
        lr_images = batch["LR_image"]
        print(f"Low resolution image shape: {lr_images.shape}")


