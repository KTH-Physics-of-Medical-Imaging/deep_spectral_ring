# import packages 
import argparse 
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm.auto import tqdm
from torch import nn
from torch import optim
from IPython.display import clear_output
import pytorch_ssim
import math 

# user defined
import models 
import utils 

def train_loop(dataloader, model, loss_fn, optimizer, lambda_1, lambda_2, vgg, patch_sz, n_patches):
    n, n_mat, h, w = next(iter(dataloader)).size()
    n_mat //= 2
    size = len(dataloader)
     
    # set up loss 
    loss_mse = nn.MSELoss() 
    loss_l1 = nn.L1Loss()
    model.train() 
    
    train_loss = 0
    tqdm_data = tqdm(dataloader)
    for idx, data in enumerate(tqdm_data):
        # set up data 
        if patch_sz is None:
            observed = data[:,0:n_mat,:,:] 
            truth = data[:,n_mat:n_mat*2,:,:] 
        else:
            batch_sz = len(data)
            #h_start = np.random.choice(np.array(range(h//2-patch_sz,h//2)), batch_sz)
            #w_start = np.random.choice(np.array(range(w//2-patch_sz,w//2)), batch_sz)
            h_start = np.random.choice(np.array(range(0,h-patch_sz)), batch_sz*n_patches)
            w_start = np.random.choice(np.array(range(0,w-patch_sz)), batch_sz*n_patches)
            observed = torch.zeros((batch_sz*n_patches,n_mat,patch_sz,patch_sz))
            truth = torch.zeros((batch_sz*n_patches,n_mat,patch_sz,patch_sz))
            k=0
            for j in range(batch_sz):
                for i in range(n_patches):
                    idx_h = torch.tensor(range(h_start[k], h_start[k]+patch_sz))
                    idx_w = torch.tensor(range(w_start[k], w_start[k]+patch_sz))
                    observed[j,:,:,:] = data[j,0:n_mat,:,:].index_select(1, idx_h).index_select(2, idx_w) 
                    truth[j,:,:,:] = data[j,n_mat:n_mat*2,:,:].index_select(1, idx_h).index_select(2, idx_w) 
                    k+=1
                    
        if torch.cuda.is_available():
            observed = observed.cuda()
            truth = truth.cuda()
            model = model.cuda()
        
        # compute prediction and loss
        pred = model(observed)
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
        
        # backpropagation 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        if (idx+1) % 100 == 0 or idx==0:
            print('Batch: [{}/{}] Loss: {:.7f}'.format(idx+1, size, loss.item()))
    train_loss /= size
    return train_loss

def test_loop(dataloader, model, loss_fn, lambda_1, lambda_2, vgg, n_samples, fig_sz, patch_sz, n_patches):
    n, n_mat, h, w = next(iter(dataloader)).size()
    n_mat //= 2
    size = len(dataloader)
        
    mu_soft = 0.0203
    mu_bone = 0.0492

    # set up loss
    loss_mse = nn.MSELoss() 
    loss_l1 = nn.L1Loss()
    loss_ssim = pytorch_ssim.SSIM(window_size = 11)   
    
    mean_loss = 0 
    mean_psnr = 0
    mean_ssim = 0

    model.eval()
        
    with torch.no_grad():
        tqdm_data = tqdm(dataloader)
        for idx, data in enumerate(tqdm_data):
            # set up data 
            if patch_sz is None:
                observed = data[:,0:n_mat,:,:] 
                truth = data[:,n_mat:n_mat*2,:,:] 
            else:
                batch_sz = len(data)
                #h_start = np.random.choice(np.array(range(h//2-patch_sz,h//2)), batch_sz)
                #w_start = np.random.choice(np.array(range(w//2-patch_sz,w//2)), batch_sz)
                h_start = np.random.choice(np.array(range(0,h-patch_sz)), batch_sz*n_patches)
                w_start = np.random.choice(np.array(range(0,w-patch_sz)), batch_sz*n_patches)
                observed = torch.zeros((batch_sz*n_patches,n_mat,patch_sz,patch_sz))
                truth = torch.zeros((batch_sz*n_patches,n_mat,patch_sz,patch_sz))
                k=0
                for j in range(batch_sz):
                    for i in range(n_patches):
                        idx_h = torch.tensor(range(h_start[k], h_start[k]+patch_sz))
                        idx_w = torch.tensor(range(w_start[k], w_start[k]+patch_sz))
                        observed[j,:,:,:] = data[j,0:n_mat,:,:].index_select(1, idx_h).index_select(2, idx_w) 
                        truth[j,:,:,:] = data[j,n_mat:n_mat*2,:,:].index_select(1, idx_h).index_select(2, idx_w) 
                        k+=1
       
            
            if torch.cuda.is_available():
                observed = observed.cuda()
                truth = truth.cuda()
                model = model.cuda()
        
            # compute prediction and loss
            pred = model(observed)
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
            # other performance metrics 
            psnr = 10 * np.log10((torch.max(truth).item()**2) / loss_mse(pred,truth).item()) 
            ssim = loss_ssim(pred, truth) 
            
            # add to sum    
            mean_loss += loss.item()
            mean_psnr += psnr 
            mean_ssim += ssim.item()
            
            # plot example output 
            if idx < n_samples:
                if n_mat == 1:
                    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True,
                        figsize=(2*fig_sz, fig_sz*3))
                    axs[0].imshow(observed[idx,0,:,:].cpu(), cmap='bone')
                    axs[1].imshow(pred[idx,0,:,:].detach().cpu(), cmap='bone')
                    axs[2].imshow(truth[idx,0,:,:].detach().cpu(), cmap='bone')
                    plt.show()
                else:
                    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True,
                        figsize=(3*fig_sz, fig_sz*3))
                    axs[0,0].imshow(observed[idx,0,:,:].cpu(), cmap='bone')
                    axs[0,1].imshow(observed[idx,1,:,:].cpu(), cmap='bone')
                    axs[0,2].imshow(mu_soft*observed[idx,0,:,:].cpu()+mu_bone*observed[idx,1,:,:].cpu(), cmap='bone')
                    axs[1,0].imshow(pred[idx,0,:,:].detach().cpu(), cmap='bone')
                    axs[1,1].imshow(pred[idx,1,:,:].detach().cpu(), cmap='bone')
                    axs[1,2].imshow(mu_soft*pred[idx,0,:,:].detach().cpu()+mu_bone*pred[idx,1,:,:].detach().cpu(), cmap='bone')
                    axs[2,0].imshow(truth[idx,0,:,:].detach().cpu(), cmap='bone')
                    axs[2,1].imshow(truth[idx,1,:,:].detach().cpu(), cmap='bone')
                    axs[2,2].imshow(mu_soft*truth[idx,0,:,:].detach().cpu()+mu_bone*truth[idx,1,:,:].detach().cpu(), cmap='bone')
                    plt.show()

    # get averages
    test_loss = mean_loss / size
    mean_psnr /= size
    mean_ssim /= size 
    
    print('Test results: \n ---------------------------')
    print('Mean test loss: {:.7f}'.format(test_loss))
    print('Mean PSNR: {:.7f}'.format(mean_psnr))
    print('Mean SSIM: {:.7f}'.format(mean_ssim))
    return torch.tensor([test_loss])

def main(args):
    # set up data 
    print('Setting up data...')
    trainloader, validloader, train_std = utils.prepare_dataloaders(args.train, args.valid, args.batch_sz, args.standardize)
    
    if args.standardize:
        torch.save(train_std, args.train + '_std.pt')
    print('Data set up done!')
           
    # set up model
    n_mat = next(iter(trainloader)).size(1)//2
    if args.net == 'resnet':
        model = models.iterative_ResNet(args.n_iter, n_mat, args.n_channels)
        save = args.net+'_'+str(args.n_iter)+'_'+str(args.n_channels) 
    elif args.net == 'unet':
        model = models.UNet(n_mat,args.init_features,norm=args.batch_norm)
        save = args.net+'_'+str(args.init_features)
    elif args.net == 'unet_alt':
        model = models.UNet_alt(n_mat,args.init_features,norm=args.batch_norm,skip=args.skip_connection, pre=args.pre_activation)
        save = args.net+'_'+str(args.init_features)
    elif args.net == 'yang':
        model = models.Generator_yang(n_mat,args.init_features)
        save = args.net+'_'+str(args.init_features)
    elif args.net == 'cycle':
        model = models.Generator_cycle(n_mat,n_mat,args.init_features)
        save = args.net+'_'+str(args.init_features)
    elif args.net == 'cycle_alt':
        model = models.Generator_cycle_alt(n_mat,n_mat,args.init_features)
        save = args.net+'_'+str(args.init_features)
    else:
        raise RuntimeError('Please provide a supported model')
    
    # to get sense of complexity 
    print('Total number of parameters:',
      sum(param.numel() for param in model.parameters()))
    
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
    
    # set up optimizer 
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.b1, args.b2))
    
    # set up saved model's name
    save_file = save+'_'+args.loss_fn+'_'+str(args.layer)+'_'+str(args.epochs) + '_' + str(n_mat) 
    if args.standardize:
        save_file += '_std'
    if args.batch_norm:
        save_file += '_bn'
    if args.skip_connection:
        save_file += '_sc'
    if args.pre_activation:
        save_file += '_pa'
    if args.patch_sz is not None:
        save_file += '_' + str(args.patch_sz) 
        save_file += '_' + str(args.n_patches)
    
    running_loss = torch.zeros((args.epochs,2))   
    # main loop 
    for epoch in range(0, args.epochs):
        print('Epoch: {}  \n ---------------------------'.format(epoch+1))
        running_loss[epoch,0] = train_loop(trainloader, model, args.loss_fn, optimizer, args.lambda_1, args.lambda_2, vgg, args.patch_sz, args.n_patches)
        clear_output()
        running_loss[epoch,1] = test_loop(validloader, model, args.loss_fn, args.lambda_1, args.lambda_2, vgg, args.n_samples, args.fig_sz, args.patch_sz, args.n_patches)
        model_state = {'epoch': epoch, 
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': running_loss
            }
        plt.plot(
            range(1,epoch+2), 
            running_loss[0:(epoch+1),0],
            label="Train"
        )
        plt.plot(
            range(1,epoch+2), 
            running_loss[0:(epoch+1),1],
            label="Val"
            )
        plt.title("Loss")
        plt.legend()
        plt.show()
        if epoch % args.log_interval == 0 and epoch != 0:
            torch.save(model_state, './results/checkpoints/'+save_file+'_'+str(epoch)+'.pt') 
             
    torch.save(model.state_dict(), './results/' + save_file  + '_' + str(args.lambda_1) + '_'  + str(args.lambda_2) \
                   + '_' + str(args.batch_sz) + '_' + args.train.split('/')[-1] + '.pt')
    torch.save(running_loss, './results/plots/' + save_file + '_' + str(args.lambda_1) + '_'  + str(args.lambda_2) \
                   + '_' + str(args.batch_sz)  + '_' + args.train.split('/')[-1] + '_plot.pt')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        )
    parser.add_argument(
        '--train',
        type = str,
        default = './data/train_kits_img',
        help = 'string indicating training set to be used',
        )
    parser.add_argument(
        '--valid',
        type = str,
        default = './data/val_kits_img',
        help = 'string indicating validation set to be used',
        )
    parser.add_argument(
        '--loss_fn',
        type = str,
        default = 'vgg16',
        help = 'string indicating loss function to be used',
        )
    parser.add_argument(
        '--batch_sz',
        type = int,
        default = 3,
        help = 'batch size.')
    parser.add_argument(
        '--patch_sz',
        type = int,
        default = None,
        help = 'patch size.')
    parser.add_argument(
        '--n_patches',
        type = int,
        default = 1,
        help = 'number of patches extracted')
    parser.add_argument(
        '--standardize',
        dest = 'standardize',
        action='store_true',
        help = 'boolean indicating data should be standardized by its channel wise standard deviation'
        )
    parser.add_argument(
        '--no-standardize',
        dest = 'standardize',
        action='store_false',
        help = 'boolean indicating data should be standardized by its channel wise standard deviation'
        )
    parser.set_defaults(standardize=False)
    parser.add_argument(
        '--layer',
        type = int,
        default = 9,
        help = 'layer used in vgg as feature extractor (Kim et al. use 23/24 in vgg16 and Yang et al. use 36 in vgg19)')
    parser.add_argument(
        '--n_iter',
        type = int,
        default = 10,
        help = 'number of iterations used in ResNet')
    parser.add_argument(
        '--n_channels',
        type = int,
        default = 32,
        help = 'number of channels used in ResNet')
    parser.add_argument(
        '--init_features',
        type = int, 
        default = 64, 
        help = 'number of initial features used in Unet')
    parser.add_argument(
        '--epochs',
        type = int, 
        default = 100, 
        help = 'number of epochs')
    parser.add_argument(
        '--net',
        type = str,
        default = 'resnet',
        help = 'string indicating network to be used (supported resnet/unet)',
        )
    parser.add_argument(
        '--batch_norm',
        dest = 'batch_norm',
        action='store_true',
        help = 'boolean indicating whether batch norm should be used in UNet'
        )
    parser.add_argument(
        '--no-batch_norm',
        dest = 'batch_norm',
        action='store_false',
        help = 'boolean indicating whether batch norm should be used in UNet'
        )
    parser.set_defaults(batch_norm=False)
    parser.add_argument(
        '--skip_connection',
        dest = 'skip_connection',
        action='store_true',
        help = 'boolean indicating whether skip connection should be used in UNet'
        )
    parser.add_argument(
        '--no-skip_connection',
        dest = 'skip_connection',
        action='store_false',
        help = 'boolean indicating whether skip_connection should be used in UNet'
        )
    parser.set_defaults(skip_connection=False)
    parser.add_argument(
        '--pre_activation',
        dest = 'pre_activation',
        action='store_true',
        help = 'boolean indicating whether pre-activation should be used in UNet'
        )
    parser.add_argument(
        '--no-pre_activation',
        dest = 'pre_activation',
        action='store_false',
        help = 'boolean indicating whether pre-activation should be used in UNet'
        )
    parser.set_defaults(pre_activation=False)
    parser.add_argument(
        '--learning_rate',
        type = float,
        default = 1e-4,
        help = 'learning rate',
        )
    parser.add_argument(
        '--b1',
        type = float,
        default = 0.5,
        help = 'b1 parameter for ADAM',
        )
    parser.add_argument(
        '--b2',
        type = float,
        default = 0.9,
        help = 'b2 parameter for ADAM',
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
        '--log_interval',
        type = int,
        default = 25,
        help = '',
        )
    parser.add_argument(
        '--n_samples',
        type = int,
        default = 3,
        help = '',
        )
    parser.add_argument(
        '--fig_sz',
        type = int,
        default = 10,
        help = '',
        )
    args = parser.parse_args()
    main(args)
