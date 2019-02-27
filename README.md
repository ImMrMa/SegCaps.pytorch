# Capsules for Object Segmentation

A barebones CUDA-enabled PyTorch implementation of the segcaps architecture in the paper "Capsules for Object Segmentation" by  [Rodney LaLonde and Ulas Bagci](https://github.com/lalonderodney/SegCaps).


## Condensed Abstract

> Convolutional neural networks (CNNs) have shown remarkable results over the last several years for a wide range of computer vision tasks. A new architecture recently introduced by Sabour et al., referred to as a capsule networks with dynamic routing, has shown great initial results for digit recognition and small image classification. Our work expands the use of capsule networks to the task of object segmentation for the first time in the literature. We extend the idea of convolutional capsules with locally-connected routing and propose the concept of deconvolutional capsules. Further, we extend the masked reconstruction to reconstruct the positive input class. The proposed convolutional-deconvolutional capsule network, called SegCaps, shows strong results for the task of object segmentation with substantial decrease in parameter space. As an example application, we applied the proposed SegCaps to segment pathological lungs from low dose CT scans and compared its accuracy and efficiency with other U-Net-based architectures. SegCaps is able to handle large image sizes (512 x 512) as opposed to baseline capsules (typically less than 32 x 32). The proposed SegCaps reduced the number of parameters of U-Net architecture by 95.4% while still providing a better segmentation accuracy.

Paper written by by Rodney LaLonde and Ulas Bagci. For more information, please check out the paper [here](https://arxiv.org/abs/1804.04241.).

## Requirements

* Python 3
* PyTorch
* TorchVision
* TorchNet
* Visdom

## Usage

**Step 1** Adjust the number of training epochs, batch sizes, etc. inside `my_train.py`.



**Step 2** Start training. you can choose your own dataset.

```console
$ python my_train.py
```


## TODO

- [ ] test the acc in the dataset

## Credits

Primarily referenced these two TensorFlow and Keras implementations:
1. [Official Keras implementation by Rodney LaLonde and Ulas Bagci](https://github.com/lalonderodney/SegCaps)
2. [TensorFlow implementation by @iwyoo](https://github.com/iwyoo/tf-SegCaps)


## Contact/Support

email:13935771565@163.com