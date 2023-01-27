# Ring Artifact Correction in Photon-Counting Spectral CT Using a Convolutional Neural Network With Spectral Loss

This repo contains code acompanying [Ring Artifact Correction in Photon-Counting Spectral CT Using a Convolutional Neural Network With Spectral Loss](https//addarxiv.com).

# Run the code
## Dependencies 
Install necessary packages via 
```sh
pip install -r requirements.txt
```

# Usage
The two main scripts are train and evaluate. 

```sh
main.py:
  --config: Training configuration.
    (default: 'None')
  --eval_folder: The folder name for storing evaluation results
    (default: 'eval')
  --mode: <train|eval|train_deq>: Running mode: train or eval or training the Flow++ variational dequantization model
  --workdir: Working directory
```

* `config` is the path to the config file. Our config files are provided in `configs/`. They are formatted according to [`ml_collections`](https://github.com/google/ml_collections) and should be quite self-explanatory.

  **Naming conventions of config files**: the name of a config file contains the following attributes:

  * dataset: Either `cifar10` or `imagenet32`
  * model: Either `ddpmpp_continuous` or `ddpmpp_deep_continuous`

*  `workdir` is the path that stores all artifacts of one experiment, like checkpoints, samples, and evaluation results.

* `eval_folder` is the name of a subfolder in `workdir` that stores all artifacts of the evaluation process, like meta checkpoints for supporting pre-emption recovery, image samples, and numpy dumps of quantitative results.

# Contact 
Dennis Hein <br />
dhein@kth.se

# Acknowledgements 
The following sources were helpful for this project:
* [Pytorch_ssim](https://github.com/Po-Hsun-Su/pytorch-ssim)
* [Implementation of UNet](https://nbviewer.org/github/amanchadha/coursera-gan-specialization/blob/main/C3%20-%20Apply%20Generative%20Adversarial%20Network%20(GAN)/Week%202/C3W2A_Assignment.ipynb)
