# Deep-Learning-Projects
This repository holds various architecture codes in Pytorch. For all the implementations, the related analysis on accuracies and losses and understanding of hyperparameters can be viewed from the Python Notebook called `main.ipynb`.

## MLP
The multi-layer perceptron is coded in Pytorch from scratch. We can extend the number of layers of this model by passing the number of neurons in the list. The code is placed in `mlp.py` file.

## ResNet18
Resnet18 is also coded in Pytorch from scratch. Resnet architectures work phenomenally on image tasks and it has achieved great importance in the convolutional neural networks community. The code is placed in `resnet18.py` file.

## MLP Mixer
MLP Mixer is also coded in Pytorch from scratch. MLP Mixer architecture achieved the competitive results in comparison to Transformers and CNN architectures. Although it does not have any positional embeddings, attention mechanism, and convolutions, however, it achieved state-of-the-art performance on huge dataset Image Tasks. It uses pure MLP layers. The code is placed in `mlpmixer.py` file.
