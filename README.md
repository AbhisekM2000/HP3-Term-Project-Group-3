# HP3-Term-Project-Group-3
**Project 3: Efficient Object Detection using Regional Convolutional Neural Networks
Objective
Convolutional Neural Networks (CNNs) have remarkable performance in machine intelligence tasks pertaining to object detection. The objective of this project is to implement an efficient CNN kernels for the Fast R-CNN architecture. The work items are as follows.**

1.Develop a custom specification file (Eg. Caffe prototxt file) for taking a pretrained CNN model in the form of a computation graph(1 student)

2.Implement the feedforward pass for CNN inferencing using Direct Convolution.(2 students)

3.Implement Optimized CUDA Convolution Kernels that uses Im2Col unrolling and GEMM.(2 students)

4.Implement Convolution using FFT computation (2 students)

5.Apart from this you may use cuBLAS routines for GEMM and Caffe's Pooling Kernel Implementation. (2 students) 

# How to run the code  

1. Open a Google Colab notebook 

2. Install protobuf using the following commands
   
   ``` 
   %%capture
   !apt-get install autoconf automake libtool curl make g++ unzip
   !git clone https://github.com/protocolbuffers/protobuf.git
   %cd protobuf
   !git submodule update --init --recursive
   !./autogen.sh
   !./configure
   !make -j8
   !sudo make install
   !sudo ldconfig 
   %cd ..

3. 
```
import shutil
import os 
%cd /content/
if os.path.isdir('/content/<Name of repo>'):
  shutil.rmtree('/content/<Name of repo>')
```

