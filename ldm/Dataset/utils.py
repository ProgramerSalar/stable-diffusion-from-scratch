import random 
from PIL import Image 


class RandomCrop:

    def __init__(self, size):

        self.size = size 


    def __call__(self, img):
        w, h = img.size 
        
        if w == self.size and h == self.size:
            return img 
        

        # Random crop coordinates 
        x = random.randint(0, w - self.size)
        y = random.randint(0, h - self.size)

        return img.crop((x, y, x + self.size, y + self.size))