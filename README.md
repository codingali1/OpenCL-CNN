# VGG16 Acceleration using OpenCL

## VGG16
[*Karen Simonyan and Andrew Zisserman, "Very deep convolutional networks for large-scale image recognition," International Conference on Learning Representations (ICLR), 2015*](https://arxiv.org/abs/1409.1556)

Due to using the CIFAR-10 dataset, the input size of this model is 32 x 32 and the output size of it is 10. It outputs 512 parameters through convolution operations.Therefore, It is made by changing the size of FC layer from 4096 to 512.

![VGG16 Architecture](./assets/architecture.jpg)

## Acceleration Method

### OpenCL
In the CNN model, acceleration through parallelism is very useful. So OpenCL is used to apply the parallelism with the GPU to this CNN model.

OpenCL is used to parallelize convolutional layers, pooling layers, and FC layers in VGG16. Next, batch processing is applied to maximize the effect of parallel processing.

### Convolutional Layer

**Transformation**

![Matrix Transformation](./assets/transformation.jpg)

**Tiling Algorithm**

![Tiling in the local memory](./assets/tiling-1.jpg)

![More work per thread](./assets/tiling-2.jpg)

## Environment
* Model: VGG16
* Dataset: CIFAR-10
* OpenCL
    * Intel GPU: OpenCL 2.1
    * AMD ROCm: OpenCL 2.0

## Performance

![Performance Graph](./assets/performance.jpg)