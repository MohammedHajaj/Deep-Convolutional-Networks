#pragma once


//=====================================================================================================================================================
//
// This file define the main parameters that control the training and inference stages of the main program which is trained using a variable 
// input size. It defines the main control parameters, the network structure, and the operationMode of the main program. Therefore this simple 
// .h can easily be used to change the settings of the training and inference stages, and change the structure of the network. All parameters 
// that are constant and do not change during training are defined as constant integers. This will prevent any function or CUDA kernel from 
// changing any of these settings unintentionally. By defining these parameters as constant integers and including this .h file in all source 
// code (.cu or .cpp) files all functions and CUDA kernels can recognize and use these parameters without being able to change them.
//
//=====================================================================================================================================================




//=====================================================================================================================================================
// NumEpoch:-  defines the number of training epochs
// BatchSize:-  defines the batch size.
// TrainSize :- defines the number of training images in the training dataset.
// TestSize :- defines the number of test images in the test set.
// ValSize :- defines the number of validation images in the validation set.
//=====================================================================================================================================================


const int NumEpoch = 6;
const int BatchSize = 20;


const int TrainSize = 6500;
const int TestSize = 6500;
const int ValSize = 6500;

//=====================================================================================================================================================
// To simplify the portioning of data into batches, we ensure that the number of images in each dataset is a multiple of the batch size.  
//=====================================================================================================================================================


const int TrainSizeM = (TrainSize / BatchSize)*BatchSize;
const int ValSizeM = (ValSize / BatchSize)*BatchSize;
const int TestSizeM = (TestSize / BatchSize)*BatchSize;



//=====================================================================================================================================================
//                                                                Network structure
//
// The network structure is defined by defining the structures of the convolutional layers, and the structure of the output layer. This is a  
// deep residual CNN, and therefore the size of the residual block (number of conv layers) is also defined. The network starts by a single 
// convolutional layer followed by multiple residual blocks followed by a single FC layer which is the output layer. Two models are defined here
// the 18-layer (17 conv layers plus 1 FC layer) and 34-layer (33 conv layers plus 1 FC layer) models from He Kaiming et al paper "Deep Residual 
// Learning for Image Recognition", with a minor difference witch is using maxpooling for resolution reduction instead of convolution with a stride
// of 2 (both options can be used will minimal changes to the program). Other (deeper) residual models can easily be defined the same way the two 
// models were defined. 
//
//=====================================================================================================================================================



//=====================================================================================================================================================
//
//                                    The structure of the convolutional and pooling layers.
//
// The main difference between this implementation and the standard CNN implementation is that this model is trained using a variable input size.
// In each new iteration (new batch of images) an input size is randomly chosen from a set of predefined sizes. As a result the sizes of the input
// and output channels of all convolutional layers change accordingly. NumWin defines the number of predefined input sizes, and the WinSize[] 
// array contains these input sizes. A variable global average pooling is used after the last convolutional layer to keep the input size to the 
// FC output layer constant, and therefore keep the number of weights in the network constant while the input size changes.
//
//=====================================================================================================================================================
// CL :- the number of conv layers. 
// InCh1 :- the number of input channels to the first conv layer. For colour RGB images InCh1 is 3.
// NumWin :- Number of predefined input sizes. 
// WinSize :- An integer array that defines a set of predefined input sizes.
// JMP :- defines the size of the residual block which is also equal to the number of consecutive conv layer that will be jumped-ahead by the residual 
// connection. The number of the convo layer should by 1 + n*JMP, where n is any integer number.  
//
// Once the number of conv layers CL is defined, the structure of each of these layers need to be defined. The basic structure of a conv layer is
// defined using 4 parameters:
//
// OutCh[k] :- the number of output channels for conv layer k.
// Kr[k] :- the square filter size of layer k is Kr[k]*Kr[k].
// PAD[k] :- the padding size for conv layer k.
// STRIDE[k] :- the convolution stride for layer k.
//
// There is 3 parameters that define the pooling layer that follows conv layer k, if such layer exist. 
// 
// PoolType[k] :- if PoolType[k]=0, this means that there is no pooling layer after conv layer k. If PoolType[k]=1, then there is a maxpooling layer 
// after conv layer k. If PoolType[k]=2, then there is an average pooling layer after conv layer k. If there a pooling layer then the following 2 
// parameters define the structure of such layer.
// Sr1[k] :- defines the pooling stride of the pooling layer that follows conv layer k.
// Sr2[k] :- defines the pooling size of the pooling layer that follows conv layer k.
//
// One note here is that the pooling stride and pooling size of the last conv layer (Sr1[CL-1] and Sr2[CL-1]) are not used because these have a variable
// value that depends on the input size of the network. As the network input changes, the value of Sr1[CL-1] and Sr2[CL-1] are set to make the last 
// pooling layer as a global pooling layer.
//=====================================================================================================================================================



//--------------------------------------------------------------------------------------
const int CL = 17;//33;//
const int InCh1 = 3;
const int JMP = 2;

const int NumWin = 6;
const int WinSize[NumWin] = { 160, 192, 224, 256, 288, 320 };

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%           The 18-layer model              %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

//                                   The parameters that define each conv layer                                           //
const int OutCh[CL] = { 64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512 };
const int       Kr[CL] = { 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 };
const int      PAD[CL] = { 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
const int   STRIDE[CL] = { 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

//                                   The parameters that define each pooling layer                                         //
const int      Sr1[CL] = { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 7 };
const int      Sr2[CL] = { 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 7 };
const int PoolType[CL] = { 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2 };

//==========================================================================================================================//


/*
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%           The 34-layer model              %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

//                                              The parameters that define each conv layer                                             //
const int OutCh[CL] = { 64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512 };
const int       Kr[CL] = { 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 };
const int      PAD[CL] = { 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
const int   STRIDE[CL] = { 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

//                                   The parameters that define each pooling layer                                         //
const int      Sr1[CL] = { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, IR1 / 32 };
const int      Sr2[CL] = { 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, IR1 / 32 };
const int PoolType[CL] = { 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2 };

//======================================================================================================================================================================================//
*/


const int wsSize = WinSize[NumWin - 1] * WinSize[NumWin - 1] * BatchSize *OutCh[0];

//=====================================================================================================================================================
//
//                                    The structure of the output layer.
// In1:- Number of input to the FC output layer. Because a global average pooling follows the last conv layer, the number of inputs to the FC output 
// layer is equal to the number of output channels in the last conv layer OutCh[CL - 1], (array index starts from zero, and therefore the index 
// of the last conv layer is k = CL - 1).
// Out1 = The number of output neurons for the output layer, which is also the number of classes when only class labels are used. 
//=====================================================================================================================================================


const int In1 = OutCh[CL - 1];
const int Out1 = 10;
const int NumClasses = Out1;


//-----------------------------------------------------------------------------------------------------------------------------------------------------
// The DecAlpha array defines the schedule that will be used to decay the learning rate. The one shown here shows the schedule used when training is 
// continued for 100 epochs, where the learning rate was decayed at epochs. This gives more flexibility to define the decay schedule than using a 
// constant interval.
//-----------------------------------------------------------------------------------------------------------------------------------------------------

const int DecAlpha[5] = { 35, 56, 72, 85, 93 };


//-----------------------------------------------------------------------------------------------------------------------------------------------------
// The RGB_GPU_SIZE is the size in bytes of the 3 RGB buffers. These buffers are divided into 2 halves, where one CPU thread loads the input images 
// into halves, and the main CPU thread consumes the input images from the other half.
//-----------------------------------------------------------------------------------------------------------------------------------------------------

const size_t RGB_GPU_SIZE = 4000000000;

//=====================================================================================================================================================
// The Operation_Mode variable that controls the mode of operation can take one of five values. Rather than using numbers to define these modes, we 
// defined constant integer parameters with names that reflect the nature of each mode.
//=====================================================================================================================================================

const int TRAIN = 1;
const int TRAIN_PLUS_INFERENCE = 2;
const int INTERRUPTED_TRAIN = 3;
const int INTERRUPTED_TRAIN_PLUS_INFERENCE = 4;
const int INFERENCE = 5;





//=====================================================================================================================================================
//                                            Defining the parameters for multi-crop Inference
//
// When using multi-crop inference, the prediction of each test image is equal to average predictions of multiple crops taken at different scales. 
// Even though the training was done using a variable input size, the inference phase is implemented using a single input size which is the median input
// size WinSize[NumWin/2]. The following parameters define the total number of crops per image, the number of scales that will be used and the number 
// of crops at each scale.
//
// Scale1: Is the smallest scale that will be used in the inference stage. The shorter side of the image will be scaled to Scale1.
// ScaleSteps :- The number of scales that will be used in the inference stage.
// ScaleInc :- scale increment is the amount of scale increment between two consecutive scaling steps. 
// EpochTs :- The number of crops per image, agian this number should be a multiple of the number of scales (ScaleSteps) to ensure that the number 
// of crops at each scale is the same.
// Scale2 :- Is the maximum scale which can be computed using Scale1, ScaleSteps, and ScaleInc.
//=====================================================================================================================================================

const int Scale1 = 280;
const int ScaleInc = 64;
const int ScaleSteps = 1;// 5;
const int EpochTs = (1 / ScaleSteps)*ScaleSteps;
const int Scale2 = Scale1 + (ScaleSteps - 1)*ScaleInc;




//=====================================================================================================================================================
// 4 constant sizes that are used to define the thread block sizes for CUDA kernels. The Choice of BLOCKSIZE2 is important for the Softmax cuda 
// kernel, and it shouldn't be less than NumClasses/8;
//=====================================================================================================================================================

const int BLOCKSIZE1 = 128;
const int BLOCKSIZE2 = 16;// 32;// 64;// 128;//64;// 1024;// 
const int BLOCKSIZE3 = 32;
const int BLOCKSIZE4 = 64;


//=====================================================================================================================================================
// The Var_Param c structure defines all convolutional layer parameters that will change as the network input size changes. These include the input channel 
// size of each conv layer IR[k]*IR[k], the output channel size of each conv layer CR[k]*CR[k], and the output channel size of each pooling layer
// SR[k]*SR[k]. It also includes other parameters (Xr[k], Yr[k]) which define the buffer sizes that hold all input and output channels for each conv
// layer k (where 0 <= k < CL).
//=====================================================================================================================================================

struct Var_Param{
	int IR[CL], CR[CL], SR[CL], Xr[CL], Yr[CL];
};

//=====================================================================================================================================================
// The Var_gridSizes c structure defines all the grid sizes of the cuda kernels that will change as the network input size changes. These include the 
// size of the pooling cuda kernel gridSizeP[k] which implements pooling layer k. The size of the cuda kernel of the second stage of the forward pass 
// of BN or the first stage of the backward pass of BN gridSizeBN2[k] for BN layer k. The size of the data augmentation cuda kernel gridSize_Crop. And 
// the size of the matrix addition cuda kernel used to pass the error signal back through the residual connection gridSizeAddYB[k] (where 0 <= k < CL).
//=====================================================================================================================================================

struct Var_gridSizes{
	dim3 gridSizeP[CL], gridSizeBN2[CL], gridSize_Crop;
	int gridSizeAddYB[CL];
};
