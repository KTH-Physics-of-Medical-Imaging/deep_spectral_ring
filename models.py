import torch
from torch.nn import functional as F
from torch import nn
from torchvision.models import vgg16, VGG16_Weights, vgg19, VGG19_Weights 
from torchvision import transforms

# based on https://nbviewer.org/github/amanchadha/coursera-gan-specialization/blob/main/C3%20-%20Apply%20Generative%20Adversarial%20Network%20(GAN)/Week%202/C3W2A_Assignment.ipynb
class ContractingStack(nn.Module):
    """
    Class for contracting stack of the UNet (encoder). Repeats a sequence of 
    conv2d and leaky relu with alternative batch norm. 
    """
    def __init__(self, in_channels, out_channels, batch_norm=False, pre_activation=False):
        super(ContractingStack,  self).__init__()   
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding = "same")
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding = "same")
        self.activation = nn.ReLU()
        if batch_norm:
            if pre_activation:
                self.bn1 = nn.BatchNorm2d(in_channels)
                self.bn2 = nn.BatchNorm2d(out_channels)
            else:
                self.bn = nn.BatchNorm2d(out_channels)
        self.batch_norm = batch_norm
        self.pre_activation = pre_activation 
    
    def forward(self, x):
        if self.pre_activation:
            if self.batch_norm:
                x = self.bn1(x)
            x = self.activation(x)
            x = self.conv1(x)
            if self.batch_norm:
                x = self.bn2(x)
            x = self.activation(x)
            x = self.conv2(x)
        else:
            x = self.conv1(x)
            if self.batch_norm:
                x = self.bn(x)
            x = self.activation(x)
            x = self.conv2(x)
            if self.batch_norm:
                x = self.bn(x)
            x = self.activation(x)
        return x

class MiddleStack(nn.Module): 
    """
    Class for middle stack of UNet. Repeats a sequence of conv2d and relu. 
    """
    def __init__(self, channels, batch_norm=False, pre_activation=False):
        super(MiddleStack,  self).__init__()   
        self.conv1 = nn.Conv2d(channels//2, channels, 3, padding = "same")
        self.conv2 = nn.Conv2d(channels, channels//2, 3, padding = "same")
        self.activation = nn.ReLU()
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(channels//2)
            self.bn2 = nn.BatchNorm2d(channels)
        self.batch_norm = batch_norm
        self.pre_activation = pre_activation 
        
    def forward(self, x):
        if self.pre_activation:
            if self.batch_norm:
                x = self.bn1(x)
            x = self.activation(x)
            x = self.conv1(x)
            if self.batch_norm:
                x = self.bn2(x)
            x = self.activation(x)
            x = self.conv2(x)
        else:
            x = self.conv1(x)
            if self.batch_norm:
                x = self.bn2(x)
            x = self.activation(x)
            x = self.conv2(x)
            if self.batch_norm:
                x = self.bn1(x)
            x = self.activation(x)
        return x 
        
class ExpandingStack(nn.Module):
    """
    Class for expanding stack of the UNet (decoder). Repeats a squence of 
    upsampling, conv2d, and contacenation followed by conv2d and relu twice, with 
    alternative batch norm.  
    """
    def __init__(self, in_channels, out_channels, batch_norm=False, pre_activation=False):
        super(ExpandingStack,  self).__init__()   
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(2*in_channels, in_channels, 3, padding = "same")
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding = "same")   
        self.activation = nn.ReLU()
        if batch_norm:
            if pre_activation:
                self.bn1 = nn.BatchNorm2d(2*in_channels)
                self.bn2 = nn.BatchNorm2d(in_channels)
            else:
                self.bn1 = nn.BatchNorm2d(in_channels)
                self.bn2 = nn.BatchNorm2d(out_channels)
        self.batch_norm = batch_norm
        self.pre_activation = pre_activation
   
    def forward(self, x, x_skip):
        if self.pre_activation:
            x = self.upsample(x)
            x = torch.cat([x, x_skip],1)
            if self.batch_norm:
                x = self.bn1(x)
            x = self.activation(x)
            x = self.conv1(x)
            if self.batch_norm:
                x = self.bn2(x)
            x = self.activation(x)
            x = self.conv2(x)
        else:
            x = self.upsample(x)
            x = torch.cat([x, x_skip],1)
            x = self.conv1(x)
            if self.batch_norm:
                x = self.bn1(x)
            x = self.activation(x)
            x = self.conv2(x)
            if self.batch_norm:
                x = self.bn2(x)
            x = self.activation(x)
        return x

class UNet_alt(nn.Module):
    '''
    Class implementing UNet with 4x contracting stacks and corresponding 
    4x expanding stacks. Upconvolution has been replaced with upsampling.
    Based on UNet used in Kim et al. 2019. 
    '''
    def __init__(self, input_channels, hidden_channels=64, norm=False, skip=False, pre=False):
        super(UNet_alt, self).__init__()
        self.first = nn.Conv2d(input_channels, hidden_channels, 1, padding="same")
        self.c1 = ContractingStack(hidden_channels,hidden_channels,batch_norm=norm,pre_activation=pre)
        self.c2 = ContractingStack(hidden_channels,hidden_channels*2,batch_norm=norm,pre_activation=pre)
        self.c3 = ContractingStack(hidden_channels*2,hidden_channels*4,batch_norm=norm,pre_activation=pre)
        self.c4 = ContractingStack(hidden_channels*4,hidden_channels*8,batch_norm=norm,pre_activation=pre)
        self.middle = MiddleStack(hidden_channels*16,batch_norm=norm,pre_activation=pre)
        self.e1 = ExpandingStack(hidden_channels*8,hidden_channels*4,batch_norm=norm,pre_activation=pre)
        self.e2 = ExpandingStack(hidden_channels*4,hidden_channels*2,batch_norm=norm,pre_activation=pre)
        self.e3 = ExpandingStack(hidden_channels*2,hidden_channels,batch_norm=norm,pre_activation=pre)
        self.e4 = ExpandingStack(hidden_channels,hidden_channels,batch_norm=norm,pre_activation=pre)
        self.final = nn.Conv2d(hidden_channels, input_channels, 1, padding="same")
        self.maxpool = nn.MaxPool2d(2)
        self.skip = skip
        
    def forward(self, x):
        n = x.clone()
        n0 = self.first(x)
        n1 = self.c1(n0)
        n2 = self.c2(self.maxpool(n1))
        n3 = self.c3(self.maxpool(n2))
        n4 = self.c4(self.maxpool(n3))
        n5 = self.middle(self.maxpool(n4))
        n6 = self.e1(n5,n4)
        n7 = self.e2(n6,n3)
        n8 = self.e3(n7,n2)
        n9 = self.e4(n8,n1)
        n10 = self.final(n9)
        
        if self.skip:
            return  n+n10
        else:
            return n10 
        
# Johnson et al. 2016:
# relu1_2 (4), relu2_2 (9), relu3_3 (16), relu4_3 (23)
# Kim et al. j = relu_j + 1
class VGG_Feature_Extractor_16(torch.nn.Module):
    def __init__(self, layer=24, n_mat = 2, requires_grad=False):
        super(VGG_Feature_Extractor_16, self).__init__()
        vgg_pretrained_features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.n_mat = n_mat
        self.slice = torch.nn.Sequential()
        for x in range(layer):
            self.slice.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        vgg_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.n_mat == 3:
            out = self.slice(vgg_transform(X))    
        elif self.n_mat == 2:
            out = torch.cat((self.slice(vgg_transform(X[:,0:1,:,:].repeat(1,3,1,1))),self.slice(vgg_transform(X[:,1:,:,:].repeat(1,3,1,1)))),dim=1)     
        else:
            out = self.slice(vgg_transform(X[:,0:1,:,:].repeat(1,3,1,1))) 
        return out
    