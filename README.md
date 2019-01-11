
This repository contains a CUDA implementation of a deep residual CNN. The code is a part of a PhD research project at Imperial College London titled "Investigating the Behavior of Deep Convolutional Networks in Image Recognition". At the final stage this repository will contain the standard network implementation, and other variations that we implemented during the research project. These variations include an implementation that can be trained using a variable input size, an implementation that uses data hierarchy by incorporating category labels, and a multitasking network that can be trained on multiple datasets.


Currently, the CNN_Standard folder conation the .cu, .h, and .cpp files that implement a standard deep residual convolutional network, the same implementaion as He et al "Deep residual learning for image recognition" with a minor difference of using maxpooling instead of using a srtide of 2 for some conv layers to perform channel resolution reduction. Both ways can be used with minimal changes to the code.  The main cuda program is named mainProgram_StandardDCN.cu, and the main cuda kernels are put in the cuda file cudaKernels.cu. All other functions are put in the Functions.cpp file. Most of the network training and inference settings are put in the ConstantSettings.h file and can easily be changed form there. 


