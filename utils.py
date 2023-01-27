import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt 
import os 
import torch.autograd as autograd
from torch.autograd import Variable

def prepare_dataloaders(train_file, valid_file,  batch_sz, standardize):
    train_data = torch.load(train_file + '.pt')
    valid_data = torch.load(valid_file + '.pt')
    
    train_std = None 
    if standardize:
        #https://discuss.pytorch.org/t/standardization-of-data/16965/4
        train_std = train_data.std((0,2,3), keepdim=True)
        train_data /= train_std
        valid_data /= train_std 
        
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_sz, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_sz, shuffle=True)
    
    return train_loader, valid_loader, train_std 

# based on function with same name from the 2017 ODL workshop
def show_image_matrix(images, titles, figsize = 10, save_path = None, **kwargs):
    n_samples, n_mat = images[0].size()[0:2]
    n_cols = len(images)
    
    fig, axs = plt.subplots(n_mat*n_samples, n_cols, sharex=True, sharey=True,
        figsize=(n_cols*figsize, figsize*n_mat*n_samples))
    idx = 0
    for i in range(n_samples):
        for mat in range(n_mat):
            col = axs[idx]
            idx += 1
            for title, img, ax in zip(titles, images, col):
                ax.set_title(title)
                plot = ax.imshow(img[i,mat,:,:], **kwargs)
                ax.set_axis_off()
                fig.colorbar(plot, ax = ax)
    if save_path is not None:
        plt.savefig(save_path+'.png')
    plt.show()

# https://stackoverflow.com/questions/13583153/how-to-zoomed-a-portion-of-image-and-insert-in-the-same-plot-in-matplotlib
# https://github.com/matplotlib/matplotlib/issues/12323/
def plot_insert(img, save=None, **kwargs):
    fig, ax = plt.subplots(figsize=[10, 10])

    #plot = ax.imshow(img, **kwargs)
    ax.imshow(img, **kwargs)

    # inset axes....
    axins = ax.inset_axes([0.55, 0.55, 0.45, 0.45]) #x_0,y_0,h,w
    axins.imshow(img, **kwargs)
    
    # sub region of the original image
    img_size = img.size()[-1]
    x1, x2, y1, y2 = (img_size//2-32), (img_size//2+32), (img_size//2-32), (img_size//2+32)
    #x1, x2, y1, y2 = 266, 316, 316, 266
    #x1, x2, y1, y2 = 241, 291, 291, 241
    #x1, x2, y1, y2 = 191, 291, 291, 191

    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])

    #ax.indicate_inset_zoom(axins, edgecolor="black")
    rectpatch, connects=ax.indicate_inset_zoom(axins,edgecolor="black")
    connects[0].set_visible(False)
    connects[1].set_visible(False)
    connects[2].set_visible(False)
    connects[3].set_visible(False)
    
    #fig.colorbar(plot, ax = ax)
    
    plt.show()
    if save is not None:
        fig.savefig(save, dpi = 120)

def ct_transform(mu, mu_water):
    return (mu-mu_water)*(1000/mu_water)

def get_mono(data):
    mu_soft = 0.0203
    mu_bone = 0.0492
    return data[:,0:1,:,:]*mu_soft+data[:,1:,:,:]*mu_bone

#https://github.com/pytorch/pytorch/issues/7415
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)