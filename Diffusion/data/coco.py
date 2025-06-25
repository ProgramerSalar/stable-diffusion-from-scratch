import os 
from PIL import Image 
import torch 
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import numpy as np 
import matplotlib.pyplot as plt

class CocoDataset(Dataset):

    def __init__(self,
                 root_dir,
                 annotation_file,
                 transform=None):
        
        
        self.root_dir = root_dir 
        self.coco = COCO(annotation_file)
        self.transform = transform

        # get all image IDs with captions 
        self.image_ids = list(self.coco.imgs.keys())

        # create caption index: list of (image_id, caption_id) pairs 
        self.caption_index = []
        for img_id in self.image_ids:
            caption_ids = self.coco.getAnnIds(imgIds=img_id)
            self.caption_index.extend([(img_id, cap_id) for cap_id in caption_ids])


    def __len__(self):
        return len(self.caption_index)
    

    def __getitem__(self, idx):

        # get Image Id and caption Id
        img_id, caption_id = self.caption_index[idx]

        # Load image 
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        # Apply image transforms 
        if self.transform:
            image = self.transform(image)

        # Load caption 
        caption_ann = self.coco.loadAnns(caption_id)[0]
        caption = caption_ann['caption']

        return image, caption
    




if __name__ == "__main__":

    from torchvision import transforms
    

    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor()
    ])


    # initial dataset 
    dataset = CocoDataset(
        root_dir = "coco_data/train2017",
        annotation_file="coco_data/captions_train2017.json",
        transform=transform
    )


    def collate_fn(batch):

        images, captions = zip(*batch)

        # stack images 
        images = torch.stack(images, 0)

        # pad captions 
        captions = captions

        return images, captions
    

    train_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )

    for images, captions in train_loader:
        # print(f"Image batch shape: {images.shape}")
        # print(f"caption batch shape: {captions}")

        image = images[0]
        print(f"what is the shape of image: {image.shape}")
        caption = captions[0]
        print(f"what is the image caption: {caption}")

        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).astype("uint8")
        image_pil = Image.fromarray(image)
        image_pil.save(f"Diffusion/data/sample/{caption}.jpg")
    

        break

    



    

        