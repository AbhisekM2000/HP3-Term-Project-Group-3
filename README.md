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

3. Setting the system path
   ```
   import shutil
   import os 
   %cd /content/
   if os.path.isdir('/content/<Name of repo>'):
   shutil.rmtree('/content/<Name of repo>')
   ```
4. Importing repo to Google Colab 
   ```
   import os
   from getpass import getpass
   import urllib
   user = input('User name: ')
   password = getpass('Password: ')
   password = urllib.parse.quote(password) # your password is converted into url format
   cmd_string = 'git clone https://{0}:{1}@<Repo link.git'.format(user, password)
   os.system(cmd_string)
   cmd_string, password = "", "" # removing the password from the variable
   ```

5. Changing the current working directory 
   ```
   %cd /content/<Repo name> 
   ```
6. Compiling the prototxt files using protobuf 
   ```
   %cd proto/
   !protoc -I=. --cpp_out=. ./network.proto
   %cd ..
   ```
   
7. Loading the pretrained models 
   ``` 
   !python ConvertToSpecification.py
   %%capture
   %cd forward/data
   !unzip MiniImageNet.zip
   %cd ../../ 
   ```
   
8. CNN forward test 
   ```
   %cd /content/H<Repo name>/forward/cnn_forward_test/  # If batch-test is to be performed we can cd to /forward/batch_test/
   !make 
   ```
   
9. Running different algorithms
   ```
   !make run_direct/ run_im2col/ run_fft # Depending upon which algorithm we want to implement our convolution with 
   ```

   
