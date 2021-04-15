# Introduction
This folder contains the main libraries for the CNN Inferencing Engine, the Data Utilities, and the Correctness Tests

# Inferencing Engine Libraries
We provide two C++ libraries to help run the forward pass of a CNN by loading a model saved in our custom specification format:

* `operations.h` 
	This contains classes for the major operations of a CNN which include:
	* Conv2D : Convolution Opertation [The convolution algorithm can be selected with the help of a parameter]
	* Pool (Pool2D) : Provides option for both MAX and AVERAGE pooling as well as support for Adaptive Pooling:
		* Adaptive Pooling is when you specify only the output size and the operation itself computes the filter sizes etc. to achieve that given any input. Thi s was added to support Pytorch's Adaptive Pooling Layer. However, the algorithm it uses is different from Pytorch and so the answers can differ. 
	* Activation : Provides both RELU and SIGMOID activations
	* Linear : Provides support for Linear/FC computations using GEMM
* `cnn_forward.h`
This libary provides two functions:
	* `loadCNNModelFromFile`: Helps load a saved model into a Network object
	* `forwardPass`: Iterates over the CNN Network object and computes the forward pass on the provided data
