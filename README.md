# VGG16 Acceleration using OpenCL

## VGG16
[*Karen Simonyan and Andrew Zisserman, "Very deep convolutional networks for large-scale image recognition," International Conference on Learning Representations (ICLR), 2015*](https://arxiv.org/abs/1409.1556)

Due to using the CIFAR-10 dataset, the input size of this model is 32 x 32 and the output size of it is 10. It outputs 512 parameters through convolution operations.Therefore, It is made by changing the size of FC layer from 4096 to 512.

![VGG16 Architecture](./assets/architecture.jpg)

## Acceleration Method

### OpenCL
In the CNN model, acceleration through parallelism is very useful. So OpenCL is used to apply the parallelism with the GPU to this model.

OpenCL is used to parallelize convolutional layers, pooling layers, and FC layers. Batch processing is then further applied to maximize the effect of parallel processing.

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
    1. Intel GPU: OpenCL 2.1
    2. AMD ROCm: OpenCL 2.0
* Specification
    1. Intel i5-8265U, DDR4 16GB, Intel UHD Graphics
    2. 

## Performance

![Performance Graph](./assets/performance.jpg)