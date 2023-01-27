# Ring Artifact Correction in Photon-Counting Spectral CT Using a Convolutional Neural Network With Spectral Loss

This repo contains code acompanying [Ring Artifact Correction in Photon-Counting Spectral CT Using a Convolutional Neural Network With Spectral Loss](https//addarxiv.com).

# Run the code
## Dependencies 
Install necessary packages via 
```sh
pip install -r requirements.txt
```

## Usage
The two main scripts are train and evaluate. To train run 
```
python train.py --FLAGS 
```
To see available flags run `python train.py -h`. Similarly, to evaluate our network we run 
```
python evaluate.py --FLAGS 
```

### Example
For instance, to train the top performing network, we ran  
```
python train.py --net unet_alt --loss_fn vgg16_l1_alt --layer 9 --lambda_1 10 --lambda_2 1 --skip_connection --batch_sz 2 --init_features 64 --patch_sz 512 --train ./data/train_kits_img --valid ./data/val_kits_img --epochs 100 --n_samples 2 --log_interval 25
```
This model is then saved as `unet_alt_64_vgg16_l1_alt_9_100_2_sc_512_10.0_1.0_2_train_kits_img` in `.\results`. To evaluate this network run 
```
python evaluate.py --net unet_alt_64_vgg16_l1_alt_9_100_2_sc_512_10.0_1.0_2_train_kits_img --loss_fn vgg16_alt --data ./data/test_kits_img --idx_print 34 --n_samples 1
```

# Contact 
Dennis Hein <br />
dhein@kth.se

# Acknowledgements 
The following sources were helpful for this project:
* [Pytorch_ssim](https://github.com/Po-Hsun-Su/pytorch-ssim)
* [Implementation of UNet](https://nbviewer.org/github/amanchadha/coursera-gan-specialization/blob/main/C3%20-%20Apply%20Generative%20Adversarial%20Network%20(GAN)/Week%202/C3W2A_Assignment.ipynb)
