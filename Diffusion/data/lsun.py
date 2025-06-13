import PIL.Image
import torch, os 
from torch.utils.data import Dataset
import PIL
from PIL import Image
from torchvision import transforms
import numpy as np 



class LSUNBase(Dataset):

    def __init__(self,
                 txt_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5):
        

        self.data_paths = txt_file
        self.data_root = data_root

        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()


        self.__length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths]
        }

        self.size = size 
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS}[interpolation]
        
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)



    def __len__(self):
        return self.__length
    

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])

        if not image.mode == "RGB":
            image = image.convert("RGB")


        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w = img.shape[0], img.shape[1]

        img = img[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]
        

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)

        example["image"] = (image / 127.5 - 1.0).astype(np.float32)

        return example
    


class LSUNBedroomsTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="Diffusion/lsun_bedroom_train.txt", 
                         data_root="Diffusion/bedroom/train",
                         **kwargs)
        

class LSUNBedroomsValidation(LSUNBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(txt_file="Diffusion/lsun_bedroom_validation.txt",
                         data_root="Diffusion/bedroom/validation",
                         flip_p=flip_p,
                         **kwargs)
        

    

if __name__ == "__main__":

    dataset = LSUNBedroomsTrain()
    # print(dataset)
    
    for d in dataset:
        print(d['image'].shape)     # (256, 256, 3)
    


        