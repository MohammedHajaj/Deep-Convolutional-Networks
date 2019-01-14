
This repository contains a CUDA implementation of a deep residual CNN. The code is a part of a PhD research project at Imperial College London titled "Investigating the Behavior of Deep Convolutional Networks in Image Recognition". At the final stage this repository will contain the standard network implementation, and other variations that we implemented during the research project. These variations include an implementation that can be trained using a variable input size, an implementation that uses data hierarchy by incorporating category labels, and a multitasking network that can be trained on multiple datasets.


The CNN_Standard folder contain the .cu, .h, and .cpp files that implement a standard deep residual convolutional network, the same implementation as He et al "Deep residual learning for image recognition" with a minor difference of using maxpooling instead of using a stride of 2 for some conv layers to perform channel resolution reduction. Both ways can be used with minimal changes to the code.  The cudnn.lib NVIDIA library is used to implement optimized versions of the conv layer. The main cuda program is named mainProgram_StandardDCN.cu, and the main cuda kernels are put in the cuda file cudaKernels.cu. All other functions are put in the Functions.cpp file. Most of the network training and inference settings are put in the ConstantSettings.h file and can easily be changed form there. 

The CNN_Variable_Input_Size folder contain the files that implement a deep residual network that can be trained using a variable input size, the cudakernels.cu file is almost identical to the cudakernels.cu used by the standard program, with the exception of the data augmentation cuda kernels which has one additional input (the randomly chosen or the selected input size). However, they can easily be made similar, and the same cudakernels.cu file can be used for both implementations. The Header.h is the same for both implementations. The ConstantSettings.h are slightly different, but they can thoughtfully be combined together in a single file that can be used for both implementations. The main differences between the standard implementation and the implementation trained using a variable input size is in the Functions.cpp file and in the main cuda programs (mainProgram_StandardDCN vs mainProgram_VariableInput). Although both implementations are using the same cuda kernels, the setup of these kernels is different. In the standard implementation each kernel was setup once using a single gird/block size per layer, while in the implementation that can be trained using a variable input size each cuda kernel was setup with multiple grid/block sizes per layer which correspond to the multiple input sizes used in the training phase. Many of the functions in Functions.cpp are still similar while the setup and initialization functions have core differences between the two implementations. Again the two files can be thoughtfully combined together in a single file that can be used for both implementations. The main cuda programs however have more differences between them in the type of initialization functions that are used, and in the way the cudnn.lib functions used to implement the conv layer were setup and called, and in the way the cuda kernels were setup and called. 

The CNN_ExtraLayerCategoryLabel folder contains the files that implements a deep residual CNN that incorporates a coarse category label in addition to the standard class label. An extra FC output layer that predicts the coarse category label is added to a standard deep residual CNN at an earlier stage in comparison to the main FC output layer that predicts the class label which is the last layer in the network. The extra FC output layer is preceded by an average pooling layer to reduce the number of weights and prevent overfitting. The location of this extra FC output layer was decided based on a t-SNE visualization of the conv layer outputs of a standard network. The cuda kernels used by this implementation are identical to those used by the standard implementation, and therefore we didn't include the cudaKernels.cu file in the main folder. Also, the header.h file is not included in the folder because it is the same as the one used by the standard program. As with the other implementations most of the program settings are put in the ConstantSettings.h, and its very similar to the file used by the standard implementation with the addition of the parameters that define the extra output layer. The Function.cpp contains all non-cuda functions and it is similar to the file used by the main program with few extra functions required to initialize and setup the extra pooling layer and the extra FC output layer. The main cuda program is named mainProgram_ExtraCategoryLabel.cu, and it added the code required to initialize, setup, and train the extra FC output layer that predicts the coarse category label of the image. The extra output layer is not used in the inference stage and therefore this implementation added no overhead when the network is used. To get the benefits of using this extra layer, similar classes need to be grouped into one coarse category and then are given the same coarse category label.
