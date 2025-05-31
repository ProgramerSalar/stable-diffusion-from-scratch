from torch.utils.data import Dataset, DataLoader
import os 
from torchvision import transforms
from PIL import Image
import torch
from torchvision.transforms import v2

class LSUNBase(Dataset):

    def __init__(self,
                 data_root):
        
        
        self.data_path = data_root
        self.image_files = [f for f in os.listdir(self.data_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.ToDtype(dtype=torch.float16, scale=False)
        ])



    def __len__(self):
        return len(self.image_files)
    

    def __getitem__(self, 
                    idx):
        
        img_name = os.path.join(self.data_path, self.image_files[idx])
        image = Image.open(img_name)
        image = self.transform(image)

        return image 
    





if __name__ == "__main__":

    data_root = "Dataset/Data/val"

    data = LSUNBase(data_root)
    dataloader = DataLoader(data, batch_size=4, shuffle=True)
    print(next(iter(dataloader)))
    
