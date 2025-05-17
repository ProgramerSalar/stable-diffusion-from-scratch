
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset 
import os 
from PIL import Image 


class ImageDataset(Dataset):

    def __init__(
            self, 
            image_folder,
            image_size = (256, 256)
    ):
        
        self.image_folder = image_folder

        # get a list of all image file paths in the folder 
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])


    def load_images(self, im_path):

        assert os.path.exists(im_path), "images paths {} does not exist".format(im_path)
        ims = []
        labels = []
        for d_name in os.listdir(im_path):
            print(d_name)


    def __len__(self):
        return len(self.image_paths)
    

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image 
    


if __name__ == "__main__":

    dataset = ImageDataset(image_folder="E:\\YouTube\\ddpm\\dataset\\cat_dog_images")

    # create a Dataloader 
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for i in dataloader:
        print(i.shape)

