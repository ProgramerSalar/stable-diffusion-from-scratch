import os 
import json 
import torch
from PIL import Image 
from torch.utils.data import Dataset, DataLoader
from .utils import RandomCrop
import numpy as np 

class COCODataset(Dataset):

    def __init__(self,
                 data_root,
                 split='train',
                 size=256,
                 interpolation="bicubic",
                 flip_p=0.5):
        

        self.size = size 
        self.split = split 
        self.flip_p = flip_p


        # path setup 
        self.image_dir = os.path.join(data_root, "images", f"{split}")
        self.ann_file = os.path.join(data_root, "annotations", f"captions_{split}.json")
        # print(self.ann_file)

        # Load annotations 
        with open(self.ann_file, "r") as f:
            annotations = json.load(f)


        # create image-caption mapping 
        self.image_data = {}
        # print(annotations["annotations"])

        for ann in annotations["annotations"]:
            # print(ann)

            image_id = ann["image_id"]

            if image_id not in self.image_data:
                
                self.image_data[image_id] = {
                    "file_name": next(i["file_name"] for i in annotations["images"]
                                    if i["id"] == image_id),

                    "captions": []
                }


            self.image_data[image_id]["captions"].append(ann["caption"])


        self.image_ids = list(self.image_data.keys())


        # Transforms 
        self.crop = RandomCrop(size)
        self.interpolation = {
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS
        }[interpolation]


    def __len__(self):
        return len(self.image_ids)


    def __getitem__(self, idx):
        
        image_id = self.image_ids[idx]
        data = self.image_data[image_id]

        # Randomly select a caption 
        caption = np.random.choice(data["captions"])

        # Load images 
        img_path = os.path.join(self.image_dir, data['file_name'])
        image = Image.open(img_path).convert("RGB")

        # Get the original image size 
        w, h = image.size 

        if w < self.size or h < self.size:
            if w < h:

                # scale based width 
                new_w = self.size 
                new_h = int(h * (self.size / w))

            else:

                # scale base height 
                new_h = self.size 
                new_w = int(w * (self.size / h))

            image = image.resize((new_w, new_h), self.interpolation)

        image = self.crop(image)

        # convert to tensor 
        image = np.array(image).astype(np.float32) / 127.5 - 1.0 
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Random horizontal flip 
        if torch.rand(1) < self.flip_p:
            image = torch.flip(image, [2])

        return {
            "image": image,
            "caption": caption
        }

            

    
        
    


    









if __name__ == "__main__":
    
    train_dataset = COCODataset(data_root="/home/manish/Desktop/stable-diffusion/stable-diffusion-from-scratch/ldm/coco_data",
                                split="train2017")
    # print(train_dataset)



    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=32,
                                  shuffle=True
                                  )
    
    # print(train_dataloader)

    for data in train_dataloader:
        image = data["image"][1]
        print(image.size())

        captions = data["caption"][1]
        print(captions)


        # show the images 
        import matplotlib.pyplot as plt 
        import numpy as np 
        import torch 

        image_np = image.permute(1, 2, 0).numpy()
        # print(image_np)

        plt.imshow(image_np)
        plt.axis('off')
        plt.title(captions)
        plt.show()

        break

        