from abc import abstractmethod
from torch.utils.data import Dataset, ConcatDataset, ChainDataset, IterableDataset

import albumentations
from PIL import Image
import numpy as np 

class Txt2ImgIterableBaseDataset(IterableDataset):

    """Define an interface to make the iterableDataset for txt2img data chainable"""


    def __init__(self,
                 num_records = 0,
                 valid_ids=None,
                 size=256):
        
        super().__init__()
        self.num_records = num_records
        self.valid_ids = valid_ids
        self.sample_ids = valid_ids
        self.size = size 

        print(f"{self.__class__.__name__} dataset contain {self.__len__()} examples.")


    def __len__(self):
        return self.num_records
    

    @abstractmethod
    def __iter__(self):
        pass 

    

class ImagePaths(Dataset):

    def __init__(self,
                 paths,
                 size=None,
                 random_crop=False,
                 labels=None):
        
        self.size = size 
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)


        if self.size is not None and self.size > 0:
            self.rescale = albumentations.SmallestMaxSize(max_size=self.size)

            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, 
                                                         width=self.size)
                
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,
                                                         width=self.size)
                
            self.preprocessor = albumentations.Compose([self.rescale,
                                                        self.cropper])
            
        else:
            self.preprocessor = lambda **kwargs: kwargs


    def __len__(self):
        return self.__length 
    

    def preprocess_image(self, image_path):
        image = Image.open(image_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)

        return image 
    

    def __getitem(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])

        for k in self.labels:
            example[k] = self.labels[k][i]

        return example
