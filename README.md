# CIFAR10-End2End-MXNet

Achieved > 94% of accurracy for CIFAR10 dataset with only 50 epochs.

By Danh Doan

## Introduction
This repository serves my purpose of implementing and experiencing different modern Convolutional Neural Networks and using them to solve the well-known [**CIFAR10**](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. MXNet is used as the main framework for Deep Learning.

> When conducting experiment with CNN architectures, I use the same training parameters to draw a comparison between various CNNs. To efficiently utilize each network, experiment with another training parameters.

All networks are trained end-to-end and are implemented from scratch. 
Besides, Batch Normalization and Drop Out layers are applied whenever possible
to increase the Accuracy and avoid Overfitting.

## Learning Rate Scheduler
1-Cycle schedule is utilized in the training procedure. The value of 1-Cycle's parameters are analyzed after performing LR range test.
In my schedule, the Triangle cycle part governs 40 epochs and the Cool-Down follows in the last 10 epochs. Depicting in the following figure:

![LRs](lr-schedule.png)



## Current Results
|Architecture | Model        | Accuracy| # Params |
|-------------|--------------|---------|----------|
| AlexNet     | AlexNet      | 89.67%  | 27.31M   |
| VGG         | VGG11        | 91.49%  | 14.50M   |
|             | VGG13        | 93.66%  | 14.68M   |
|             | VGG16        | 93.42%  | 20M      |
|             | VGG19        | 92.87%  | 25.31M   |
| ResNet      | ResNet18     | 92.36%  | 11.19M   |
|             | ResNet34     | 92.39%  | 21.31M   |
|             | ResNet50     | 91.86%  | 23.59M   |
|             | ResNet101    | 91.52%  | 42.66M   |
|             | ResNet152    | 91.30%  | 58.38M   |
| DenseNet    | DenseNet121  | 91.86%  | 3.27M    |
|             | DenseNet161  | 92.69%  | 12.30M   |
|             | DenseNet169  | 91.31%  | 5.99M    |
|             | DenseNet201  | 91.61%  | 8.5M     |
| GoogleNet   | GoogleNet    | 86.91%  | 6.07M    |
| Inception   | Inception V3 | 94.25%  | 19.33M   |


## Training History
* AlexNet:

![AlexNet](history/alexnet-acc-0.8967.png)

* VGG13:

![VGG13](history/vgg13-acc-0.9366.png)

* ResNet34:

![ResNet34](history/resnet34-acc-0.9239.png)

* DenseNet161:

![DenseNet161](history/densenet161-acc-0.9269.png)

* Inception-V3:

![Inception-V3](history/inceptionv3-acc-0.9425.png)


## Latest Updates
* 2019, Aug 20:
  * Apply **1-Cycle** for Learning Rate Scheduler [[paper]](https://arxiv.org/pdf/1708.07120.pdf)
  * Re-train all models with only 50 epochs and still achieve comparable accurracy or over higher

* 2019, Aug 16:
  * Apply LR scheduler built-in module from MXNet

* 2019, Aug 13:
  * Implement and Test with all **Inception V3** architectures [[paper]](https://arxiv.org/pdf/1512.00567.pdf)

* 2019, Aug 12:
  * Implement and Test with all **GoogleNet** architectures [[paper]](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf)

* 2019, Aug 10:
  * Implement and Test with all **DenseNet** architectures [[paper]](https://arxiv.org/abs/1608.06993)

* 2019, Aug 8:
  * Implement and Test with all **ResNet** architectures [[paper]](https://arxiv.org/abs/1512.03385)
  * Implement and Test with all **AlexNet** architectures [[paper]](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

* 2019, Aug 7:
	* Set up the training and test program
	* Implement and Test with all **VGG** architectures [[paper]](https://arxiv.org/abs/1409.1556)

## Installation
* Install MXNet framework and GluonCV toolkit
	* For CPU only:
	
		`pip install mxnet gluoncv`
	
	* For GPUs
		
		`pip install mxnet-cu90 gluoncv`
    	> Change to match with CUDA version. `mxnet-cu100` if CUDA 10.0 is installed
	
