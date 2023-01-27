# Ring Artifact Correction in Photon-Counting Spectral CT Using a Convolutional Neural Network With Spectral Loss

This repo contains code acompanying [Ring Artifact Correction in Photon-Counting Spectral CT Using a Convolutional Neural Network With Spectral Loss](https//addarxiv.com).

# Run the code
## Dependencies 
Install necessary packages via 
```sh
pip install -r requirements.txt
```

# Usage
The two main scripts are train and evaluate. To train run 
```
python train.py --FLAGS 
```
To see available flags run `python train.py -h`. Similarly, to evaluate our network we run 
```
python evaluate.py --FLAGS 
```

## Example
To train the top performing network run 
```
python train.py --net unet_alt --loss_fn vgg16_l1_alt --layer 9 --lambda_1 10 --lambda_2 1 --skip_connection --batch_sz 2 --init_features 64 --patch_sz 512 --train ./data/train_kits_img --valid ./data/val_kits_img --epochs 100 --n_samples 2 --log_interval 25
```

This model is then saved as `resnet_3_32_100.pth` in `.\results`. To evaluate the performance of this network, while plotting 6 samples and saving the predictions as a .mat files to be passed to the image reconstruction routine we run
```
python evaluate.py --net resnet_3_32_100 --n_samples 6 --save_mat 
```

# Contact 
Dennis Hein <br />
dhein@kth.se

# Acknowledgements 
The following sources were helpful for this project:
* [Pytorch_ssim](https://github.com/Po-Hsun-Su/pytorch-ssim)
* [Implementation of UNet](https://nbviewer.org/github/amanchadha/coursera-gan-specialization/blob/main/C3%20-%20Apply%20Generative%20Adversarial%20Network%20(GAN)/Week%202/C3W2A_Assignment.ipynb)
