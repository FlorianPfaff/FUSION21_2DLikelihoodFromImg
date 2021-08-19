# Deep Likelihood Learning for 2-D Orientation Estimation Using a Fourier Filter

This repository is the official implementation of [Deep Likelihood Learning for 2-D Orientation Estimation Using a Fourier Filter](https://isas.iar.kit.edu/pdf/FUSION21_Pfaff.pdf).

## Requirements
Check out the git repository and the submodules for example via
```
git clone --recurse-submodules https://github.com/KIT-ISAS/FUSION21_2DLikelihoodFromImg.git
```

The network is trained in PyTorch and can be used with Python. However, the likelihoods are used for filters that are implemented in Matlab in our evaluation. For this, we call Python from Matlab.

To setup a conda environment that installs everything required for training or using the network run

```setup
conda env create -f fusion21-2dlikelihood-cpu.yml
```
Note that this installation does not include GPU support. Install PyTorch with GPU support to enable training on the GPU. Our training code uses [TorchDirectional](https://github.com/KIT-ISAS/TorchDirectional), which is included as a submodule. We also build upon a [modified implementation of LeNet-5](https://github.com/activatedgeek/LeNet-5).

For the evaluation, Matlab is required. The code was tested on Matlab 2021a. The code requires [libDirectional](https://github.com/KIT-ISAS/libDirectional), which requires certain Matlab toolboxes. The code also uses the [FilterEvaluationFramework](https://github.com/FlorianPfaff/FilterEvaluationFramework). Both libDirectional and the FilterEvaluationFramework are included as submodules for your convenience.

## Data Set

You can download the training and validation data used in the paper at [Zenodo](https://doi.org/10.5281/zenodo.5234115). You can also generate data yourself using the Matlab script generateDataset in the folder DatasetGeneration. 

Our tests were performed using a tracking scenario. All image data used for test purposes are generated fresh in the evaluation script. The data is directly passed on the model and is not stored to the disk.

## Training

First activate the conda environment via

```
conda activate fusion21-2dlikelihood-cpu
```

Then, go the folder Training and run the training via

```train
python likelihood_from_img.py
```
to train the network with the default parameters. You can directly specify some important parameters directly from the command line. For example, to train the network for to output real Fourier coefficients for 100 epochs with a learning rate of 0.0006, use

```trainAdvanced
python likelihood_from_img.py --n_coeffs 101 --epochs 100 --use_real True --lr 0.0006
```

## Evaluation

To perform the evaluation described in the paper, go the folder Evaluation and run

```evaluation
evaluateForAllShapes
```
using Matlab. The additional calculation for the figures in the papers can be done by running 
```plots
paperPlots
```

## Pre-trained Models

The pretrained model (<1 MB) is directly included in this repository in the folder PretrainedModel.

## License

My code is under GPL 3.0. Code by other authors listed as requirements is linked to as submodules and no code  by other people is directly contained in this repository. All submodules are under their respective licenses.

## Contributing
Open an issue or write me an email to <pfaff@kit.edu> if you have suggestions or experience issues.