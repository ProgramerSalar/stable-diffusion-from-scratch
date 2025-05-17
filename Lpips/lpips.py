import torch 
import torch.nn as nn 
from torchvision import models
from collections import namedtuple
from utils import get_ckpt_path

class Vgg16(nn.Module):

    """ 
    A model that extract features from different layers of pretrained VGG model.
    """

    def __init__(self,
                 requires_grad=False,
                 pretrained=True):
        

        super().__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features

        # Define different slices of the vgg16 model to extract features 
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        for x in range(4):
            self.slice1.add_module(name=str(x),
                                   module=vgg_pretrained_features[x])
            
        for x in range(4, 9):
            self.slice2.add_module(name=str(x),
                                   module=vgg_pretrained_features[x])
            
        for x in range(9, 16):
            self.slice3.add_module(name=str(x),
                                   module=vgg_pretrained_features[x])
            
        for x in range(16, 23):
            self.slice4.add_module(name=str(x),
                                   module=vgg_pretrained_features[x])
            
        for x in range(23, 30):
            self.slice5.add_module(name=str(x),
                                   module=vgg_pretrained_features[x])
            

        # Freeze parameters if requires_grad is False 
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False 




    def forward(self, x):

        """Passes the input through different slices and collects features map."""

        h = self.slice1(x)
        print(f"Slice 1 output shape: {h.shape}")   # [B, 64, H, W]
        h_relu1_2 = h 

        h = self.slice2(h)
        print(f"Slice 2 output shape: {h.shape}")   # [B, 128, H/2, W/2]
        h_relu2_2 = h 

        h = self.slice3(h)
        print(f"Slice 3 output shape: {h.shape}")   # [B, 256, H/4, H/4]
        h_relu3_3 = h 

        h = self.slice4(h)
        print(f"Slice 4 output shape: {h.shape}")  # [B, 512, H/8, H/8]
        h_relu4_4 = h 

        h = self.slice5(h)
        print(f"Slice 5 output shape: {h.shape}")  # [B, 512, H/16, H/16]
        h_relu5_5 = h 

        vgg_output = namedtuple(typename="VggOutputs",
                                field_names=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_4', 'relu5_5']
                                )
        out = vgg_output(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_4, h_relu5_5)

        return out
    

class ScalingLayer(nn.Module):

    def __init__(self):
        super().__init__()

        # Shift tensor used for normalization 
        self.register_buffer(name='shift',
                             tensor=torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        
        # Scale tensor used for normalization 
        self.register_buffer(name = 'scale', 
                             tensor = torch.Tensor([.458, .448, .450])[None, :, None, None])
        

    def forward(self, input):
        return (input - self.shift) / self.scale 


class NetLinLayer(nn.Module):

    def __init__(self,
                 chn_in, 
                 chn_out=1,
                 use_dropout=False):
        

        super().__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(in_channels=chn_in,
                             out_channels=chn_out, 
                             kernel_size=1, 
                             stride=1, 
                             padding=0,
                             bias=False), ]
        self.model = nn.Sequential(*layers)



        



class LPIPS(nn.Module):

    """ 
    Learned Perceptual Image Path Similarity (LPIPS) model,
    used for comparing perceptual differences between images.
    """

    def __init__(self,
                 use_dropout=True):
        
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.channels = [64, 128, 256, 512, 512]

        # Initialzed layer for features comparison
        self.net = Vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(chn_in=self.channels[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(chn_in=self.channels[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(chn_in=self.channels[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(chn_in=self.channels[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(chn_in=self.channels[4], use_dropout=use_dropout)
        self.load_from_pretrained()

        # Freeze model parameters 
        for param in self.parameters():
            param.requires_grad = False 


    def load_from_pretrained(self, name="vgg_lpips"):

        ckpt = get_ckpt_path(name, root="E:\\Coding-For-YouTube\\Lpips")
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print(f"Loaded pretrained LPIPS loss from {ckpt}")

    
    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        
        model = cls()
        ckpt = get_ckpt_path(name, root="E:\\Coding-For-YouTube\\Lpips")
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        return model
    


    def forward(self, input, target):

        """ 
        Computes LPIPS similarity between input and target images.
        """

        # used to scaling layer to normalized the input_tensor and target_tensor
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))

        # passes both images through a pre-trained network 
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        
        for kk in range(len(self.channels)):
            # Normalize the features (normalize tensor) likely scale them to unit length
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            # Compute the squared differences between the normalized features of `input` and `target`
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2 


        lines = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        # Applies a learned linear layer `lins[kk]` to weight to importance at the level 
        # Spatially average the weight differences (reducing each features map to a single value.)
        res = [spatial_average(lines[kk].model(diffs[kk]), keep_dim=True) for kk in range(len(self.channels))]

        # Sums the scores from all features levels to produce the final LPIPS distances.
        val = res[0]
        for l in range(1, len(self.channels)):
            val += res[l]

        return val 
    




def normalize_tensor(x, eps=1e-10):

    """
    Normalize a tensor along channel dimension
    """

    norm_factor = torch.sqrt(torch.sum(x **2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keep_dim=True):

    """ 
    Compute spatial mean over height and width dimensions.
    """

    return x.mean([2, 3], keepdim=keep_dim)

if __name__ == "__main__":
    # vgg = Vgg16().eval()
    # print(vgg)


    # create sample images 
    img1 = torch.randn(1, 3, 256, 256)
    img2 = torch.randn(1, 3, 256, 256)

    # compute perceptual distance 
    lpips = LPIPS().eval()
    distance = lpips(img1, img2)
    print(f"Perceptual distance: {distance.item(): .4f}")
    
# -----------------------------------------------------------------------------

    # Load the pre-trained VGG-Lpips model 
    lpips_model = lpips.from_pretrained('vgg_lpips')

    distance = lpips_model(img1, img2)
    print(f"LPIPS distance: {distance.item():.3f}")






