# Import packages
import argparse 
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
import scipy.io as io
import pytorch_ssim
import os 

# user defined
import models 
import utils 

def test(dataloader, model, loss_fn, train_std, n_samples, lambda_1, lambda_2, net, val_set, vgg, save_mat,idx_print):
    n_mat = next(iter(dataloader)).size(1)//2
    size = len(dataloader.dataset)
    
    # set up different loss configurations 
    loss_mse = nn.MSELoss() 
    loss_l1 = nn.L1Loss()
    loss_ssim = pytorch_ssim.SSIM(window_size = 11)   
    
    loss_list = [] 
    psnr_list = []
    ssim_list = []
    psnr_list_id = []
    ssim_list_id = []
    psnr_list_mono = []
    ssim_list_mono = [] 
    
    mu_soft_40 = 0.0282
    mu_bone_40 = 0.1244
    mu_soft_70 = 0.0203
    mu_bone_70 = 0.0492
    mu_soft_100 = 0.0179
    mu_bone_100 = 0.0355
    mu_water_40 = 0.0265        
    mu_water_70 = 0.0193
    mu_water_100 =  0.0170
    ww = 400
    wl = 40
    
    model.eval()
        
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            observed = data[:,0:n_mat,:,:] 
            truth = data[:,n_mat:n_mat*2,:,:] 
            # set up data 
            if train_std is not None: 
                for i in range(0,n_mat):
                    observed[:,i,:,:] /= train_std[:,i,:,:]
            if torch.cuda.is_available():
                observed = observed.cuda()
                truth = truth.cuda()
                model = model.cuda()
        
            # compute prediction and loss
            pred = model(observed)
            if train_std is not None:
                for i in range(0,n_mat):
                    pred[:,i,:,:] *= train_std[:,i,:,:].cuda()
                    observed[:,i,:,:] *= train_std[:,i,:,:].cuda()
            if loss_fn == 'mse':
                loss = loss_mse(pred, truth)
            elif loss_fn == 'mse_l1':
                loss = lambda_1*loss_mse(pred, truth) + lambda_2*loss_l1(pred,truth)
            elif loss_fn == 'l1':
                loss = loss_l1(pred, truth)
            elif loss_fn == 'vgg16' or loss_fn == 'vgg19':
                loss = loss_mse(vgg(pred), vgg(truth))
            elif loss_fn == 'vgg16_alt':
                loss = loss_mse(vgg(utils.get_mono(pred)),vgg(utils.get_mono(truth)))
            elif loss_fn == 'vgg16_mse' or loss_fn == 'vgg19_mse':
                loss = lambda_1*loss_mse(vgg(pred), vgg(truth))+lambda_2*loss_mse(pred,truth)
            elif loss_fn == 'vgg16_l1' or loss_fn == 'vgg19_l1':
                loss = lambda_1*loss_mse(vgg(pred), vgg(truth))+lambda_2*loss_l1(pred,truth)
            elif loss_fn == 'vgg16_l1_alt' or loss_fn =='vgg19_l1_alt':
                loss = lambda_1*loss_mse(vgg(utils.get_mono(pred)),vgg(utils.get_mono(truth)))+lambda_2*loss_l1(pred,truth)
            else:
                raise RuntimeError('Please provide a supported loss function')
            
            truth_70 =  truth[:,0:1,:,:]*mu_soft_70 + truth[:,1:,:,:]*mu_bone_70
            pred_70 = pred[:,0:1,:,:]*mu_soft_70 + pred[:,1:,:,:]*mu_bone_70
            
            truth_40 =  truth[:,0:1,:,:]*mu_soft_40 + truth[:,1:,:,:]*mu_bone_40
            pred_40 = pred[:,0:1,:,:]*mu_soft_40 + pred[:,1:,:,:]*mu_bone_40
            
            truth_100 =  truth[:,0:1,:,:]*mu_soft_100 + truth[:,1:,:,:]*mu_bone_100
            pred_100 = pred[:,0:1,:,:]*mu_soft_100 + pred[:,1:,:,:]*mu_bone_100
            
            # other performance metrics 
            psnr = 10 * np.log10((torch.max(truth).item()**2) / loss_mse(pred,truth).item()) 
            psnr_id = 10 * np.log10((torch.max(truth).item()**2) / loss_mse(observed,truth).item()) 
            ssim = loss_ssim(pred, truth) 
            ssim_id = loss_ssim(observed, truth)
            
            psnr_40 = 10 * np.log10((torch.max(truth_40).item()**2) / loss_mse(pred_40,truth_40).item())
            psnr_70 = 10 * np.log10((torch.max(truth_70).item()**2) / loss_mse(pred_70,truth_70).item())
            psnr_100 = 10 * np.log10((torch.max(truth_100).item()**2) / loss_mse(pred_100,truth_100).item())
            psnr_mono = (psnr_40+psnr_70+psnr_100)/3
            
            ssim_40 = loss_ssim(pred_40, truth_40) 
            ssim_70 = loss_ssim(pred_70, truth_70) 
            ssim_100 = loss_ssim(pred_100, truth_100) 
            ssim_mono = (ssim_40+ssim_70+ssim_100)/3
            
            # add to sum / list     
            loss_list += [loss.item()]
            psnr_list += [psnr]
            psnr_list_id += [psnr_id]
            psnr_list_mono += [psnr_mono]
            ssim_list += [ssim.item()] 
            ssim_list_id += [ssim_id.item()] 
            ssim_list_mono += [ssim_mono.item()]
     
            if idx < (idx_print+n_samples) and idx >= idx_print:                               
                if n_mat == 1:
                    utils.plot_insert(truth[0,0,:,:].cpu(),cmap='bone')
                    utils.plot_insert(observed[0,0,:,:].cpu(),cmap='bone')
                    utils.plot_insert(pred[0,0,:,:].cpu(),cmap='bone')
                    utils.plot_insert(pred[0,0,:,:].cpu()-truth[0,0,:,:].cpu(),cmap='bone')
                else:
                    virtual_truth_70 =  truth[:,0:1,:,:]*mu_soft_70 + truth[:,1:,:,:]*mu_bone_70
                    virtual_obs_70 = observed[:,0:1,:,:]*mu_soft_70 + observed[:,1:,:,:]*mu_bone_70
                    virtual_pred_70 = pred[:,0:1,:,:]*mu_soft_70 + pred[:,1:,:,:]*mu_bone_70
                    truth = torch.cat((truth, virtual_truth_70),1)
                    observed = torch.cat((observed, virtual_obs_70),1)
                    pred = torch.cat((pred, virtual_pred_70),1)
                    
                    virtual_truth_40 =  truth[:,0:1,:,:]*mu_soft_40 + truth[:,1:,:,:]*mu_bone_40
                    virtual_obs_40 = observed[:,0:1,:,:]*mu_soft_40 + observed[:,1:,:,:]*mu_bone_40
                    virtual_pred_40 = pred[:,0:1,:,:]*mu_soft_40 + pred[:,1:,:,:]*mu_bone_40
                    
                    virtual_truth_100 =  truth[:,0:1,:,:]*mu_soft_100 + truth[:,1:,:,:]*mu_bone_100
                    virtual_obs_100 = observed[:,0:1,:,:]*mu_soft_100 + observed[:,1:,:,:]*mu_bone_100
                    virtual_pred_100 = pred[:,0:1,:,:]*mu_soft_100 + pred[:,1:,:,:]*mu_bone_100
                    
                    utils.plot_insert(truth[0,1,:,:].cpu(),cmap='bone')
                    utils.plot_insert(observed[0,1,:,:].cpu(),cmap='bone')
                    utils.plot_insert(pred[0,1,:,:].cpu(),cmap='bone')
                    utils.plot_insert(pred[0,1,:,:].cpu()-truth[0,1,:,:].cpu(),cmap='bone')
                     
                    utils.plot_insert(utils.ct_transform(virtual_truth_40[0,0,:,:].cpu(),mu_water_40),cmap='bone',clim = [wl-ww/2,wl+ww/2])
                    utils.plot_insert(utils.ct_transform(virtual_obs_40[0,0,:,:].cpu(),mu_water_40),cmap='bone',clim = [wl-ww/2,wl+ww/2])
                    utils.plot_insert(utils.ct_transform(virtual_pred_40[0,0,:,:].cpu(),mu_water_40),cmap='bone',clim = [wl-ww/2,wl+ww/2])
                    utils.plot_insert(utils.ct_transform(virtual_pred_40[0,0,:,:].cpu(),mu_water_40)-utils.ct_transform(virtual_truth_40[0,0,:,:].cpu(),mu_water_40),cmap='bone',clim = [(wl-ww/2)*1e-2,(wl+ww/2)*1e-2])
                    
                    utils.plot_insert(utils.ct_transform(virtual_truth_70[0,0,:,:].cpu(),mu_water_70),cmap='bone',clim = [wl-ww/2,wl+ww/2])
                    utils.plot_insert(utils.ct_transform(virtual_obs_70[0,0,:,:].cpu(),mu_water_70),cmap = 'bone',clim = [wl-ww/2,wl+ww/2])
                    utils.plot_insert(utils.ct_transform(virtual_pred_70[0,0,:,:].cpu(),mu_water_70),cmap='bone',clim = [wl-ww/2,wl+ww/2])
                    utils.plot_insert(utils.ct_transform(virtual_pred_70[0,0,:,:].cpu(),mu_water_70)-utils.ct_transform(virtual_truth_70[0,0,:,:].cpu(),mu_water_70),cmap='bone',clim = [(wl-ww/2)*1e-2,(wl+ww/2)*1e-2])
                    
                    utils.plot_insert(utils.ct_transform(virtual_truth_100[0,0,:,:].cpu(),mu_water_100),cmap='bone',clim = [wl-ww/2,wl+ww/2])
                    utils.plot_insert(utils.ct_transform(virtual_obs_100[0,0,:,:].cpu(),mu_water_100),cmap='bone',clim = [wl-ww/2,wl+ww/2])
                    utils.plot_insert(utils.ct_transform(virtual_pred_100[0,0,:,:].cpu(),mu_water_100),cmap='bone',clim = [wl-ww/2,wl+ww/2])
                    utils.plot_insert(utils.ct_transform(virtual_pred_100[0,0,:,:].cpu(),mu_water_100)-utils.ct_transform(virtual_truth_100[0,0,:,:].cpu(),mu_water_100),cmap='bone',clim = [(wl-ww/2)*1e-2,(wl+ww/2)*1e-2])
                                                                
                if save_mat:
                    save_dir = './results/imgs/'+net+'_' + val_set.split('/')[-1]+'/'
                    try:
                        os.mkdir(save_dir)
                    except FileExistsError:
                        pass 
                
                    pred_save = pred.cpu()
                    obs_save = observed.cpu()
                    truth_save = truth.cpu()
    
                    # Permute to fit order in matlab routine 
                    #pred_save = pred_save.permute(0,3,2,1)
                    #obs_save = obs_save.permute(0,3,2,1)
                    #truth_save = truth_save.permute(0,3,2,1)
    
                    # Move to numpy 
                    pred_save = pred_save.numpy()
                    obs_save = obs_save.numpy()
                    truth_save = truth_save.numpy()
                
                    # set up dictionaries 
                    mat_dict = {'prediction' : pred_save}
                    mat_dict.update({'observed' : obs_save})
                    mat_dict.update({'truth' : truth_save})
                
                    # save as .mat   
                    io.savemat(save_dir+str(idx)+'_data.mat',mat_dict)
            
    print('Test results: basis images \n ---------------------------')
    print('Mean loss: {:.4f} '.format(np.mean(loss_list)))
    print('Std loss: {:.4f} '.format(np.std(loss_list)))
    print('Mean PSNR: {:.4f} '.format(np.mean(psnr_list)))
    print('Std PSNR: {:.4f} '.format(np.std(psnr_list)))
    print('Mean SSIM: {:.4f} '.format(np.mean(ssim_list)))
    print('Std SSIM: {:.4f} '.format(np.std(ssim_list)))
    print('Mean PSNR (id): {:.4f} '.format(np.mean(psnr_list_id)))
    print('Std PSNR (id): {:.4f} '.format(np.std(psnr_list_id)))
    print('Mean SSIM (id): {:.4f} '.format(np.mean(ssim_list_id)))
    print('Std SSIM (id): {:.4f} '.format(np.std(ssim_list_id)))
    print('Mean PSNR (avg. mono): {:.4f} '.format(np.mean(psnr_list_mono)))
    print('Std PSNR (avg. mono): {:.4f} '.format(np.std(psnr_list_mono)))
    print('Mean SSIM (avg. mono): {:.4f} '.format(np.mean(ssim_list_mono)))
    print('Std SSIM (avg. mono): {:.4f} '.format(np.std(ssim_list_mono)))
  
def main(args):
    # raise error if no model is given
    if args.net is None:
        raise RuntimeError('Please provide a model')
    
    # load data 
    data = torch.load(args.data + '.pt')
    n_mat = data.size(1)//2
    
    # normalization applied to training set 
    train_std = None
    if args.std is not None:
        train_std = torch.load( './data/' + args.std + '.pt') 
                  
    # set up trained model 
    string_split = args.net.split('_')
    batch_norm = False
    skip_connection = False
    pre_activation = False
    if 'bn' in string_split:
        batch_norm = True
    if 'sc' in string_split:
        skip_connection = True
    if 'pa' in string_split:
        pre_activation = True
    if string_split[0] == 'unet':
        if string_split[1] == 'alt':
            model = models.UNet_alt(n_mat, int(string_split[2]),norm=batch_norm,skip=skip_connection, pre=pre_activation) 
        else:
            model = models.UNet(n_mat, int(string_split[1]),norm=batch_norm) 
    elif string_split[0] == 'yang':
        model = models.Generator_yang(n_mat, 32)
    elif string_split[0] == 'resnet':
        model = models.iterative_ResNet(int(string_split[1]), n_mat, int(string_split[2]))
    elif string_split[0] == 'cycle':
        if string_split[1] == 'alt':
            model = models.Generator_cycle_alt(n_mat, n_mat, int(string_split[2]))
        else:
            model = models.Generator_cycle(n_mat, n_mat, int(string_split[1]))
            
    else:
        raise RuntimeError('Please provide a supported model')
    
    # set up perceptual loss 
    vgg = None 
    if args.loss_fn=='vgg16_l1_alt' or args.loss_fn =='vgg16_alt': 
        vgg = models.VGG_Feature_Extractor_16(layer=args.layer, n_mat=1,requires_grad=False)
    elif args.loss_fn=='vgg19_l1_alt': 
        vgg = models.VGG_Feature_Extractor_16(layer=args.layer, n_mat=1,requires_grad=False)
    elif args.loss_fn.split('_')[0] == 'vgg16':
        vgg = models.VGG_Feature_Extractor_16(layer=args.layer,n_mat=n_mat,requires_grad=False)
    elif args.loss_fn.split('_')[0] == 'vgg19':
        vgg = models.VGG_Feature_Extractor_19(layer=args.layer,n_mat=n_mat,requires_grad=False)

    # move to cuda 
    if args.loss_fn.split('_')[0] == 'vgg16' or args.loss_fn.split('_')[0] == 'vgg19':
        vgg = vgg.cuda()
    
    model.load_state_dict(torch.load('./results/' + args.net + '.pt',map_location='cpu'))
        
    # set up dataloader 
    #data = (data - torch.min(data))/(torch.max(data)-torch.min(data)) 
    dataloader = torch.utils.data.DataLoader(data, batch_size=args.batch_sz, shuffle=False)

    # get performance metrics 
    test(dataloader, model, args.loss_fn, train_std, args.n_samples, args.lambda_1, args.lambda_2, args.net, args.data, vgg, args.save_mat,args.idx_print)
    
# for booleans see: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        )
    parser.add_argument(
        '--data',
        type = str,
        default = './data/test_kits_img',
        help = 'string indicating dataset to be used',
        )
    parser.add_argument(
        '--batch_sz',
        type = int,
        default = 1,
        help = 'batch size'
        )
    parser.add_argument(
        '--loss_fn',
        type = str,
        default = 'mse',
        help = 'string indicating loss function to be used',
        )
    parser.add_argument(
        '--layer',
        type = int,
        default = 24,
        help = 'layer used in vgg as feature extractor (Kim et al. use 23/24 in vgg16 and Yang et al. use 36 in vgg19)')
    parser.add_argument(
        '--std',
        type = str,
        default = None,
        help = 'string indicating standard deviation of training data',
        )
    parser.add_argument(
        '--net',
        type = str,
        default = None,
        help = 'string indicating network to be used (default: None)',
        )
    parser.add_argument(
        '--n_weights',
        type = int,
        default = 20,
        help = 'number of weights used in data "augmentation". Note that var_plot will be fairly slow for large n_weights.'
        )
    parser.add_argument(
        '--n_samples',
        type = int,
        default = 3,
        help = 'number of samples to be displayed. Note that this is only for plotting. The performance metrics use the entire dataset.'
        )
    parser.add_argument(
        '--idx_print',
        type = int,
        default = 0,
        help = 'print particular slice (assuming test set)'
        )
    parser.add_argument(
        '--lambda_1',
        type = float,
        default = 1,
        help = 'Weight given to first loss objective',
        )
    parser.add_argument(
        '--lambda_2',
        type = float,
        default = 1,
        help = 'Weigh given to second loss objective',
        )
    parser.add_argument(
        '--save_mat',
        dest = 'save_mat',
        action='store_true',
        help = 'boolean indicating whether n_samples slices should be saved as .mat'
        )
    parser.add_argument(
        '--no-save_mat',
        dest = 'save_mat',
        action='store_false',
        help = 'boolean indicating whether n_samples slices should be saved as .mat'
        )
    parser.set_defaults(save_mat=True)
    args = parser.parse_args()
    main(args)
    
