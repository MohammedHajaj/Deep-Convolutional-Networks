

#include"Header.h"
#include"cudaKernels.cu"


//============================================================================================================================================================
//
// This CUDA program implements a deep residual CNN that can be trained on multiple homogeneous classification tasks. The program breaks a large dataset that
// have a number of classes (NumClasses) into multiple tasks (NumTasks) with each task having an equal number of classes (NmuClassesTask = NumClasses/NumTasks)
// The program can easily be modified to allow tasks to have different number of classes by defining an array NmuClassesTask[NumTasks] to hold the number of 
// classes per tasks. ImageNet is used to construct these tasks and most of the classes have the same number of images (1300), and constructing tasks with 
// equal number of classes makes them equal in size and diversity which can help the multitask network. Even if the tasks have different number of classes the
// reshuffle function can still give each task the same amount of exposure to the network, although that might give more weight to the classes of smaller tasks.
// Based on the class label number, the program will allocate the first NumClassesTask classes for the first task, the second NumClassesTask classes for the  
// second task, and so on. We did that because the class labels were assigned randomly to the classes of the main dataset when the dataset was constructed. 
// Again this can easily be changed to construct each task by randomly choosing a NumClassesTask classes from the main dataset if the classes of the main 
// dataset was not randomized. The program can also be easily modified to read from multiple datasets, rather than dividing the main dataset into multiple
// datasets. However, constructing multiple dataset by dividing one main large dataset such as ImageNet allows us to change the number of tasks (NumTasks) 
// simply by changing the number of classes per task (NumClassesTask), and therefore allows us to carry out more experiments with different number of tasks 
// without the need to construct and save new datasets.
//
// At the training phase the reshuffle function will construct batches from all tasks one after the other and allow the network to circulate through the tasks
// in a round robin fashion. At the last epoch the network calculates a fixed set of BN means and variances for each task using only the classes that belong 
// to that task. At the inference stage the network uses the corresponding set of means and variances for each task. The conv layer is implemented using the 
// cudnn.lib library provided by NVIDIA which implements different optimized versions of the conv layer that runs on fast NVIDIA GPUs. This program is implemented
// using Microsoft Visual Studio 2013. 
//
//============================================================================================================================================================


int Operation_Mode = TRAIN_PLUS_INFERENCE;


//============================================================================================================================================================
// Most of the parameters that define the settings of the program and the network structure are constant in the sense that they do not change during training.
// All such parameters are defined in the ConstantSettings.h file. The settings of the program (such as the number of tasks NumTasks)can easily be changed by 
// changing those parameters defined as constant integers. The few floating point parameters that are defined here are the learning rate lr, the L2 weight 
// decay parameter lmda. The learning rate is divided over the batch size to remind us to increase the learning rate with the same amount we increase the 
// batch size to maintain the same effective learning rate. By dividing the learning rate by the batch size here, the RMSprop SGD update equation implicitly
// divides lmda by BatchSize also, and therefore the effective lmda is 0.0001 in this case when the batch size is 128.
//============================================================================================================================================================


float lr = 0.005f / float(BatchSize);
float lmda = 0.01f;


//============================================================================================================================================================
// The program assumes that the main dataset (training data, validation data, and test data) files that contain the input images and image labels are stored in
// a folder where the the full directory path plus folder name is stored in the DataFloder[] char array. The program assumes that there are 5 files to store each
// of the train, validation and test sets. 3 files to store the 3 input RGB channels, one file to store the image labels, and one file to store the image dimensions
// (height and width). In the file that stores the image dimensions the heights for all images should be stored first and then the widths for all images should 
// stored second. The image labels are stored as integer values that reflect the class number. The RGB channels are stored as unsigned char buffers where each
// pixel can take an integer value between 0 and 255. The tasks are constructed by dividing the main training, validation and test sets.
//
// The 5 file names used for the training dataset are :-
//
// trainRed.txt :- to hold the Red input channels for the training images.
// trainGreen.txt :- to hold the Green input channels for the training images.
// trainBlue.txt :- to hold the Blue input channels for the training images.
// trainlabels.txt :- to hold the image labels for the training images.
// CoordinatesTr.txt :- to hold the image dimensions for the training images.
//
// The 5 file names used for the validation dataset are :-
//
// valRed.txt :- to hold the Red input channels for the validation images.
// valGreen.txt :- to hold the Green input channels for the validation images.
// valBlue.txt :- to hold the Blue input channels for the validation images.
// vallabels.txt :- to hold the image labels for the validation images.
// CoordinatesVal.txt :- to hold the image dimensions for the validation images.
//
// If the validation set is also used as a test set as is usually the case for datasets such as ImageNet, then here is no need to include a third dataset defined
// as the test set. If there is a separate test set then the 5 file names used for the test dataset are :-
//
// testRed.txt :- to hold the Red input channels for the test images.
// testGreen.txt :- to hold the Green input channels for the test images.
// testBlue.txt :- to hold the Blue input channels for the test images.
// testlabels.txt :- to hold the image labels for the test images.
// CoordinatesTs.txt :- to hold the image dimensions for the test images.
//
// These file names can be changed of course in the  ReadFile() CPU thread that reads training and validation images to the main memory, at the start of the 
// inference phase in the main program when the input test image are read, and in the InitializeTrainingData() function that reads and initializes the buffers
// that hold the images labels and dimensions. However the way the 5 file5 setup shouldn't be changed without changing the code to accommodate such changes.
// Translating the .JPEG images into 3 files to hold the 3 RGB channels was done using MATLAB code which makes easier it to inspect the generated files. 
// Changing the code to directly read .JPEG can be done by adding extra code or by changing the functions that read the input images form the disk drive.
////============================================================================================================================================================


char DataFloder[] = "C:/Data100Rand/";


//============================================================================================================================================================
// The main program runs two CPU threads, where one reads the data from disk to memory and the other consumes data from memory. The following few parameters 
// are used to synchronize between the 2 (producer-consumer) CPU threads.
//============================================================================================================================================================


int slot = 0;
mutex mu;
std::condition_variable not_empty, not_full;


//============================================================================================================================================================
// Few auxiliary variables plus the random generator used to generate random numbers from different distributions.
//============================================================================================================================================================

float ErrorT, ErrorV, MSE = 0.0f;
int CountT, CountV;

random_device rd;
mt19937 gen(rd());

//============================================================================================================================================================
// The ReadFile() function implements the CPU thread that reads data from the SSD drive to the main memory. The function uses the mutex variable mu, the 
// condition variable not_empty and not_full, and the integer variable slot to synchronize with the main CPU thread that consumes the data from the main 
// memory and use it to train the network. The image dimensions and labels are much smaller in size in comparison to the size of images themselves (the
// size of RGB input channels), and therefore they are read before the start of the training phase. This function only reads the input RGB channels one 
// segment at a time and put it in one half of the RGB buffers while the main CPU thread consumes the data from the other half. This thread reads the input
// RGB channels for the training dataset and for the validation dataset. 
//============================================================================================================================================================

void ReadFile(unsigned char *Red, unsigned char *Green, unsigned char *Blue, size_t *PartSize, int NParts, int VParts, int i0)
{
	FILE *in1, *in2, *in3, *in4, *in5, *in6;

	char FileName[128];

	strcpy_s(FileName, 128, DataFloder); strcat_s(FileName, 128, "trainRed.txt");
	fopen_s(&in1, FileName, "rb");
	strcpy_s(FileName, 128, DataFloder); strcat_s(FileName, 128, "trainGreen.txt");
	fopen_s(&in2, FileName, "rb");
	strcpy_s(FileName, 128, DataFloder); strcat_s(FileName, 128, "trainBlue.txt");
	fopen_s(&in3, FileName, "rb");
	strcpy_s(FileName, 128, DataFloder); strcat_s(FileName, 128, "valRed.txt");
	fopen_s(&in4, FileName, "rb");
	strcpy_s(FileName, 128, DataFloder); strcat_s(FileName, 128, "valGreen.txt");
	fopen_s(&in5, FileName, "rb");
	strcpy_s(FileName, 128, DataFloder); strcat_s(FileName, 128, "valBlue.txt");
	fopen_s(&in6, FileName, "rb");



	for (int i = i0; i < NParts*NumEpoch&& MSE < 100; i++)
	{
		int p = i%NParts;
		size_t altr = (i % 2)*(RGB_GPU_SIZE / 2);


		if (p == 0)
		{
			fseek(in1, long(0), SEEK_SET);
			fseek(in2, long(0), SEEK_SET);
			fseek(in3, long(0), SEEK_SET);
			fseek(in4, long(0), SEEK_SET);
			fseek(in5, long(0), SEEK_SET);
			fseek(in6, long(0), SEEK_SET);
		}
		//================================================================================

		unique_lock<mutex> locker(mu);

		while (slot == 2 && MSE < 100)
			not_full.wait(locker);

		if (p < NParts - VParts)
		{
			size_t numread1 = fread(Red + altr, sizeof(unsigned char), PartSize[p], in1);
			size_t numread2 = fread(Green + altr, sizeof(unsigned char), PartSize[p], in2);
			size_t numread3 = fread(Blue + altr, sizeof(unsigned char), PartSize[p], in3);
		}
		else
		{
			size_t numread4 = fread(Red + altr, sizeof(unsigned char), PartSize[p], in4);
			size_t numread5 = fread(Green + altr, sizeof(unsigned char), PartSize[p], in5);
			size_t numread6 = fread(Blue + altr, sizeof(unsigned char), PartSize[p], in6);
		}

		slot++;

		not_empty.notify_one();
		locker.unlock();

		//================================================================================
	}

	fclose(in1);
	fclose(in2);
	fclose(in3);
	fclose(in4);
	fclose(in5);
	fclose(in6);

}

//============================================================================================================================================================

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

//            The  Main Propgram 

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


int main()
{

	//=======================================================================================================================================================
	//
	// First, different types of return values for different cuda libraries and functions are defined. We used these return values lightly on the fly for 
	// debugging but they can be used intensively to spot the cause for error. Because the cuda code for implementing deep CNNs is complex, all 
	// cuda kernels were implemented and tested individually, and therefore the main program didn't require intensive code for exception and 
	// debugging.
	// 
	//=======================================================================================================================================================

	cudaError_t cudaStatus;
	cublasStatus_t blasstatus;
	curandStatus_t curandStatus;

	cudnnStatus_t dnnStatus;
	cudnnHandle_t dnnHandle;
	dnnStatus = cudnnCreate(&dnnHandle);

	cublasHandle_t handle;
	cublasCreate(&handle);

	cudaEvent_t start, stop;
	cudaStatus = cudaEventCreate(&start);
	cudaStatus = cudaEventCreate(&stop);



	//=======================================================================================================================================================
	//
	//         Initializing a cuda random generator.
	// 
	//=======================================================================================================================================================

	curandGenerator_t cuda_gen;
	curandStatus = curandCreateGenerator(&cuda_gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandStatus = curandSetPseudoRandomGeneratorSeed(cuda_gen, rd());



	//=======================================================================================================================================================	
	//
	// Initializing output files to store the network parameters, the per epoch results, and the BN fixed means and variances that are used in the inference stage. 
	//
	//=======================================================================================================================================================

	FILE *outResults, *outParam, *outMeansVariances, *outParamCopy, *inParam, *inMeansVariances;

	outResults = fopen("Results.txt", "wb");

	//=======================================================================================================================================================
	// If the Operation_Mode involves training then three output file are opened to store data. The outParam file pointer is used to store the parameters
	// of the trained network. The outParamCopy file pointer is used to store a copy of the network parameters (including a running average of the derivative
	// per parameters used during training by RMSprop) multiple times during training in case training is interrupted, training can be resumed from the last
	// stored copy, which is very importatnt when training lasts for days and weeks. The outMeansVariances file pointer is used to store a set of BN fixed means 
	// and variances for each task which can be used in the inference stage.
	//=======================================================================================================================================================

	if (Operation_Mode == TRAIN || Operation_Mode == INTERRUPTED_TRAIN || Operation_Mode == TRAIN_PLUS_INFERENCE || Operation_Mode == INTERRUPTED_TRAIN_PLUS_INFERENCE)
	{
		outParam = fopen("Output_param_final.txt", "wb");
		outParamCopy = fopen("Output_param_copy.txt", "wb");
		outMeansVariances = fopen("Output_MeansVariances.txt", "wb");
	}

	//=======================================================================================================================================================
	// In case training was interrupted the inParam file pointer is used to read the last stored copy of the network paramters, and resume training from 
	// that point.
	//=======================================================================================================================================================

	if (Operation_Mode == INTERRUPTED_TRAIN || Operation_Mode == INTERRUPTED_TRAIN_PLUS_INFERENCE)
	{
		inParam = fopen("Input_param_copy.txt", "rb");
	}

	//=======================================================================================================================================================
	// If the Operation_Mode includes inference only, then the inParam file pointer is used to load the parameters of a previously trained network, and the 	
	// inMeansVariances file pointer is used to load the fixed BN means and variances produced by the same network previously.  
	//=======================================================================================================================================================

	if (Operation_Mode == INFERENCE)
	{
		inParam = fopen("Output_param_final.txt", "rb");
		inMeansVariances = fopen("Output_MeansVariances.txt", "rb");
	}




	//=======================================================================================================================================================
	//
	// The input images are stored in three buffers Red, Green, and Blue, each storing one of the 3 RGB input channels for all images. Each buffer has 
	// a predefined size of RGB_GPU_SIZE bytes. Each buffer is divided into two equally sized parts where one CPU thread fills data into one part while 
	// another CPU thread (that executes the main program) consumes the data from the other part. Therefore the training data is divided into multiple 
	// segment if the total size of each of the RGB channels exceeds RGB_GPU_SIZE/2. The function InitializeTrainingData() allocates nd initializes the 
	// following data buffers.
	//
	//
	// d_HeightTr and d_WidthTr:-  GPU buffers stores the image heights and widths for all training\validation images and later for test images. 
	// d_StartTr:-  GPU buffer stores the first memory/file byte location for each image.
	// d_T :-       GPU buffer stores the image labels. 
	// NParts :- number of data segments in case the RGB total channel sizes for training+validation or test images exceeds RGB_GPU_SIZE/2.
	// VParts :- number of data segments in case the RGB total channel sizes for the validation images exceeds RGB_GPU_SIZE / 2.
	// PStart :-    buffer stores the image index of the first image in each data segment.  
	// StartPart :- buffer stores the first byte location for each data segment.
	// PartSize :-  buffer stores the size (number of bytes) for each data segemnt. 
	// 
	//=======================================================================================================================================================

	unsigned char *Red, *Green, *Blue;

	cudaStatus = cudaHostAlloc(&Red, sizeof(unsigned char)*RGB_GPU_SIZE, cudaHostAllocDefault);
	cudaStatus = cudaHostAlloc(&Green, sizeof(unsigned char)*RGB_GPU_SIZE, cudaHostAllocDefault);
	cudaStatus = cudaHostAlloc(&Blue, sizeof(unsigned char)*RGB_GPU_SIZE, cudaHostAllocDefault);


	unsigned int *d_HeightTr, *d_WidthTr;
	int *d_T, *PStart, NParts, VParts;
	size_t *d_StartTr, *PartSize, *StartPart;


	InitializeTrainingData(&d_HeightTr, &d_WidthTr, &d_StartTr, &d_T, &PStart, &PartSize, &StartPart, &NParts, &VParts, TrainSize + ValSize);




	//=======================================================================================================================================================
	//
	// The InitializeConvLayerParam() initializes the following variables which define the structure of the convolutional layers.
	//
	// CL :-    The number of convolutional layers.
	// Xr[i], Xc[i] :- define the total size of the input channels for layer i, total input channel size of all images in the batch is Xr[i]×Xc[i].
	// Yr[i], Yc[i] :- define the total size of the output channels for layer i, total output channel size of all images in the batch is Yr[i]×Yc[i].
	// WSize[i] :-     Number of weights in convolutional layer i.
	// IR[i] :-        the height/width of a single square input channel for convolutional layer i.
	// CR[i] :-        the height/width of a single square output channel before maxpooling for convolutional layer i.
	// SR[i] :-        the height/width of a single square output channel after maxpooling for convolutional layer i.
	// InCh[i] :-      number of input channels for convolutional layer i.
	// OutCh[i] :-     number of output channels for convolutional layer i.
	// In1:-           number on inputs for the FC output layer.
	// TEMP:-          temporary main memory buffer used to swap data between main memory and GPU memory.
	//
	//=======================================================================================================================================================

	int Xr[CL], Xc[CL], Yr[CL], Yc[CL], WSize[CL], IR[CL], CR[CL], SR[CL], InCh[CL];
	float *TEMP;

	InitializeConvLayerParam(IR, CR, SR, InCh, Xr, Xc, Yr, Yc, WSize, &TEMP);




	//=======================================================================================================================================================
	//
	// The InitialzeCuDNN() initializes the following cudnn.lib tensor, filter, and convolution descriptors and convolution algorithms for the convolutional layers.
	//
	// Desc_X[i] :-    Tensor descriptor of the input channels of convolutional layer i.
	// Desc_Y[i] :-    Tensor descriptor of the output channels of convolutional layer i.
	// Desc_Xs[i] :-   Tensor descriptor of the input channels of the residual connection at layer i.
	// Desc_Ys[i] :-   Tensor descriptor of the output channels of the residual connection at layer i.
	// Desc_W[i] :-       Filter descriptor of convolutional layer i.
	// Desc_Ws[i] :-      Filter descriptor of the residual connection at layer i.
	// Conv_Desc[i] :-    Convolution descriptor for convolutional layer i.
	// Conv_s_Desc[i] :-  Convolution descriptor for the residual connection at layer i.
	// FwdAlg[i] :-    Convolution algorithm to propagate the signal forward through convolutional layer i.
	// BwdDataAlg :-   Convolution algorithm to propagate the error signal backward through convolutional layer i.
	// BwdFilterAlg :- Convolution algorithm to propagate the error signal for the purpose of updating the parameters of convolutional layer i
	//
	//=======================================================================================================================================================

	cudnnTensorDescriptor_t Desc_X[CL], Desc_Y[CL], Desc_Xs[CL], Desc_Ys[CL];
	cudnnFilterDescriptor_t Desc_W[CL], Desc_Ws[CL];
	cudnnConvolutionDescriptor_t Conv_Desc[CL], Conv_s_Desc;

	cudnnConvolutionFwdAlgo_t FwdAlg[CL];
	cudnnConvolutionBwdDataAlgo_t BwdDataAlg[CL];
	cudnnConvolutionBwdFilterAlgo_t BwdFilterAlg[CL];

	InitialzeCuDNN(Desc_X, Desc_Y, Desc_W, Conv_Desc, Desc_Xs, Desc_Ys, Desc_Ws, &Conv_s_Desc, FwdAlg, BwdDataAlg, BwdFilterAlg, InCh, IR, SR);




	//=======================================================================================================================================================
	//
	// The AllocateParamGPUMemory() function allocates the main GPU memory required to store the parameters of all weight layers, and their current 
	// derivatives, and the running averages of the squared derivatives of all trainable parameters. The definitions of these variables is provided in 
	// the xxx.cu file that includes the implementation of the AllocateParamGPUMemory() function. 

	float **d_W, **d_V, **d_DW, **d_Ws, **d_Vs, **d_DWs, **d_X, **d_Y, **d_Param, **d_DParam, **d_ParamV, **d_SMU, **d_Derv;
	float *d_WF, *d_VF, *d_DWF, *d_YF, *d_Yv;
	int **d_Indx, WsSize[CL];
	bool **d_F;

	AllocateParamGPUMemory(&d_W, &d_V, &d_DW, &d_Ws, &d_Vs, &d_DWs, &d_X, &d_Y, &d_Param, &d_DParam, &d_ParamV, &d_SMU, &d_Derv, &d_WF, &d_VF, &d_DWF, &d_YF, &d_Yv, &d_Indx, &d_F, WSize, WsSize, Xr, Xc, Yr, Yc, SR, In1);




	//=======================================================================================================================================================	
	// The main dataset was already divided into parts or segments (by the InitializeTrainingData() function) that can fit in the memory, and the 
	// DividDatasetIntoMultipleTasks() function uses the int ***Indx parameter to divide each data part between the tasks by separating the indices of 
	// images that belongs to different tasks and storing them in different matrices. This function also calculates the number of images per task per
	// data part in the int **CountImgeTask parameter. Also, this function calculates the number of training and validation images that will be used 
	// in the training phase (TrainSizeM and ValSizeM) which can be slightly different from the actual numbers (TrainSize, ValSize) if different tasks
	// has different number of training and validation images. This is because all tasks are given the same amount of exposure to the network. the Indx 
	// and CountImgeTask will be used later by the ReshuffleImagesMT() function to shuffle and construct batches form different tasks in a round robin 
	// fashion. 
	//=======================================================================================================================================================	

	int TrainSizeM, ValSizeM;
	int *d_Indx1, *Indx1, ***Indx, **CountImgeTask;

	DividDatasetIntoMultipleTasks(&d_Indx1, &Indx1, &Indx, &CountImgeTask, &TrainSizeM, &ValSizeM, d_T, PStart, NParts, VParts);




	//=======================================================================================================================================================	
	//
	// The AllocateAuxiliaryGPUMemory() function allocates additional GPU memory required to store some intermediate results, and store data required 
	// by few cuda kernels. The d_SMUs buffer used to store the fixed BN means and variances that will be used in the inference stage will store multiple
	// sets for multiple tasks instead of storing a single set for a single task. The The definitions of the other variables is provided in the Function.cpp 
	// file that includes the implementation of the AllocateAuxiliaryGPUMemory() function. 
	//
	//=======================================================================================================================================================	

	float **d_SMUs, *d_YY, *d_Y0, *d_ws, *d_mse, *d_count, *d_rand1, *d_randRGB, *d_Cropf;
	unsigned int *d_Crop;

	AllocateAuxiliaryGPUMemory(&d_SMUs, &d_YY, &d_Y0, &d_ws, &d_rand1, &d_randRGB, &d_Cropf, &d_Crop, &d_mse, &d_count, PStart, Yr, Yc, NParts, TrainSizeM);




	//=======================================================================================================================================================
	//
	// The InitializeCudaKernels() function initializes the following cuda kernel variables that define the thread structure of all cuda kernels used
	// in this program. In general there are the block size which defines the number and structure of the threads inside an individual theard block, 
	// and there is the grid size which defines the number and structure of these blocks of threads. Each of the theardsize or gridsize can be defined 
	// as a single integer variable or as a dim3 variable which is a cuda structure that contains 3 integer values. Most cuda kernels use similar block
	// sizes, and usually the grid size defines the structure of the cuda kernel. The following grid sizes define the thread structure of the main cuda 
	// kernels used in this program. 
	//
	// gridSizeP[i] :-      the grid size for the maxpooling kernel of convolutional layer i.
	// gridSizeBN1[i] :-    the grid size for the first stage of the forward pass and the second stage of the backward pass kernel for BN of convolutional layer i.
	// gridSizeBN11[i] :-   the grid size for the first stage of the forward pass kernel for BN of convolutional layer i.
	// gridSizeBN2[i] :-    the grid size for the second stage of the forward pass and the first stage of the backward pass kernel for BN of convolutional layer i.
	// gridSizeAddA[i] :-   the grid size for the cuda kernel used to update the BN trainable parameters at layer i. 
	// gridSizeRGB :-       the grid size for the cuda kernel used to generate stochastic values used for colour augmentation.
	// gridSize_Crop :-     the grid size for the data augmentation kernel.
	// gridSizePA :-        the grid size for the global average pooling used after the last convolutional layer. 
	// gridSizeAddYB[i] :-  the grid size used by the AddMatrix kernel which adds two matrices with equal sizes.
	// gridSizeAddWs[i] :-  the grid size for the cuda kernel used to update the weights of the residual connection at layer i. 
	// gridSizeAddW[i] :-   the grid size for the cuda kernel used to update the weights of convolutional layer i. 
	// gridSizeAddWF :-     the grid size for the cuda kernel used to update the weights of FC output layer.
	//
	//=======================================================================================================================================================	

	dim3  gridSizeP[CL], gridSizeBN1[CL], gridSizeBN11[CL], gridSizeBN2[CL], gridSizeAddA[CL], gridSizeRGB, gridSize_Crop, gridSizePA;

	int gridSizeAddYB[CL], gridSizeAddWs[CL], gridSizeAddW[CL], gridSizeAddWF;


	InitializeCudaKernels(gridSizeP, gridSizeBN1, gridSizeBN11, gridSizeBN2, gridSizeAddA, &gridSizeRGB, &gridSize_Crop, &gridSizePA, gridSizeAddYB, gridSizeAddWs, gridSizeAddW, &gridSizeAddWF, SR, Yr, Yc, WSize, WsSize, TrainSizeM);




	//=======================================================================================================================================================
	// blockSize1 to 4 are thread block sizes defined to be used by the cuda kernels. alpha1 and alpha2 are used by cublas.lib and cudnn.lib functions.
	// wsZise is the work-space memory size required by the cudnn.lib convolution algorithms.
	//=======================================================================================================================================================	

	dim3 blockSize1(BLOCKSIZE1, 1, 1); dim3 blockSize2(BLOCKSIZE2, 1, 1); dim3 blockSize3(BLOCKSIZE3, 1, 1); dim3 blockSize4(BLOCKSIZE4, 1, 1);

	float alpha1 = 1.0f, beta1 = 0.0f, alpha2 = 1.0f, beta2 = 1.0f;

	int wsSize = ISize1*BatchSize*OutCh[0];




	//=======================================================================================================================================================	
	// The Iter integer varaible is the training iteration count which will be used by RMSprop.The kk integer variable is used to reference the DecAlpha[] array
	// which is used to define the intervals where the training rate will be decayed.
	//
	//
	// The ParameterInitialization() function initializes the trainable parameters of the network. It initializes d_W the weights of all conv layers, 
	// d_WF the weights of the FC output layer, and d_Ws the weights of the conv layers of the residual connections using Kamming initialization.
	// It also initializes d_Param the trainable parameters of the batch normalization layers.
	//=======================================================================================================================================================	

	int i0 = 0, kk0 = 0, Iter0 = 0;

	//=======================================================================================================================================================	
	// If training starts from random weights then the ParameterInitialization() function is used to initialize the network weights using Kaiming initialization.
	//=======================================================================================================================================================	

	if (Operation_Mode == TRAIN || Operation_Mode == TRAIN_PLUS_INFERENCE)
	{
		ParameterInitialization(d_W, d_Ws, d_WF, d_Param, TEMP, WSize, WsSize, In1);
	}

	//=======================================================================================================================================================	
	// When carrying inference only, the network parameters are initialized by loading the parameters of a previously trained network using the inParam
	// file pointer.
	//=======================================================================================================================================================	

	if (Operation_Mode == INFERENCE)
	{
		ReloadParameters1(inParam, TEMP, d_W, d_Ws, d_WF, d_Param, WSize, WsSize, In1);
	}

	//=======================================================================================================================================================	
	// If training was interuppted, then the network parameters (including the running averages of the derivatives) are initailzed by loading the last 
	// saved copy before interupption happened.
	//=======================================================================================================================================================	

	if (Operation_Mode == INTERRUPTED_TRAIN || Operation_Mode == INTERRUPTED_TRAIN_PLUS_INFERENCE)
	{
		ReloadParameters2(inParam, TEMP, d_W, d_V, d_Ws, d_Vs, d_WF, d_VF, d_Param, d_ParamV, WSize, WsSize, In1, &i0, &Iter0, &kk0, &lr);
	}

	int Iter = Iter0, kk = kk0;
	int jAugm;

	//=======================================================================================================================================================	
	//
	//                                    Start of the training phase
	// Two CPU threads are executed in a producer-consumer relationship, where the ReadFile reads the input images from the disk drive to the main memeory
	// and the main CPU threads consumes the data stored by the ReadFile thread, and use it to train the network. We find that for the 34 Layer residual 
	// network such implementation elementates the latency of reading from a mid range 500 MB/s sata SSD drive.
	//=======================================================================================================================================================	


	if (Operation_Mode == TRAIN || Operation_Mode == INTERRUPTED_TRAIN || Operation_Mode == TRAIN_PLUS_INFERENCE || Operation_Mode == INTERRUPTED_TRAIN_PLUS_INFERENCE)
	{
		//=======================================================================================================================================================	
		// ReadFile is the second CPU thread that plays the producer part in the producer-consumer relationship with the main thread (program). The ReadFile
		// thread reads the input channels into one half of the 3 RGB buffers, and the main program consumes the avaliable data in the other half. 
		//=======================================================================================================================================================	

		thread t1(ReadFile, Red, Green, Blue, PartSize, NParts, VParts, i0);
		this_thread::sleep_for(chrono::seconds(10));

		//---------------------------------------------------------------------------------------------------------

		cudaStatus = cudaEventRecord(start, 0);

		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		//
		//                                                           Main Forloop main CPU thread
		//
		// This is the main training for loop where CUDA kernels and functions are called to train a multitask deep residual CNN on an NVIDIA GPU. The main 
		// differences between this implementation and the single-task implementation are in the reshuffle function ReshuffleImagesMT() and in the way the 
		// BatchNormForwardT1b() cuda kernel is used. The ReshuffleImagesMT() function shuffles and arranges batches from different tasks in a round robin
		// fashion. The BatchNormForwardT1b() cuda kernel calculates multiple sets of fixed BN means and variances for multiple tasks in the last epoch, 
		// instead of calculating a single set for a single task.
		//
		// The first part of the for loop implements the forward pass, and the second part implements the backward pass, the backpropagation of the error
		// signal signal to update the trainable parameters of all layers. Because the training+validation input RGB channels are divided into (NParts)
		// segments to fit into main memory this for loop is executed Epoch*NParts times where Epoch is the number of the training epoches.
		//
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		for (int i = i0; (i < NParts*NumEpoch) && MSE < 100; i++)
		{
			//=======================================================================================================================================================	
			// p:- is the current data segment index, epoch:- is the current epoch index, alter:- makes the main CPU thread alternate between the two halfs of the 
			// of the GPU buffers that contain the RGB input channels (producer-consumer relationship).
			//=======================================================================================================================================================	

			int p = i%NParts;
			int epoch = i / NParts;
			size_t altr = (i % 2)*(RGB_GPU_SIZE / 2);

			//=======================================================================================================================================================	
			// The curandGenerate() curand.lib function generates 3 integer random values per image, 2 used to decide the cropping postion and 1 to decide on
			// horizontal flipping. curandGenerateUniform() curand.lib function generates 2 uinform random values between 0 and 1 for each images, one used to 
			// decide the amount of scaling, and the other used to decide on the amount of aspect ratio augmenation. The curandGenerateNormal() generates 3 random
			// variables per image to be used by the RGBrandPCA() cuda kernel to generate 3 stochastic numbers to be added to the 3 RGB input channels for 
			// colour augmentation.
			//=======================================================================================================================================================	

			if (p == 0)
			{
				curandStatus = curandGenerate(cuda_gen, d_Crop, 3 * TrainSizeM);
				curandStatus = curandGenerateUniform(cuda_gen, d_Cropf, 2 * TrainSizeM);
				curandStatus = curandGenerateNormal(cuda_gen, d_rand1, 3 * TrainSizeM, 0.0f, 0.1f);
				RGBrandPCA<1> << <gridSizeRGB, blockSize1 >> >(d_randRGB, d_rand1, TrainSizeM);
				jAugm = 0;
			}

			//=======================================================================================================================================================	
			// The ReshuffleImagesMT() function is more complex than the simple ReshuffleImages() used to shuffle the images in the standard single-task implementation.
			// The ReshuffleImagesMT() uses the Indx and CountImgeTask buffers which store the image indices and image count for each task and for each data segment
			// to reshuffle the images of each task separately. Then it will generate an index matrix (Indx1) for each data part or segment so that batches from different
			// tasks are stored one after the other in a round robin fashion. The training for loop for a single data part (for(j)) will then use the image indices in 
			// the Indx1 buffer to circulate through the tasks in a round robin fashion. 
			//=======================================================================================================================================================	

			int partsize;
			ReshuffleImagesMT(&partsize, Indx1, Indx, CountImgeTask, PStart, p);
			cudaStatus = cudaMemcpyAsync(d_Indx1, Indx1, sizeof(int) * partsize, cudaMemcpyHostToDevice);


			cudaStatus = cudaMemsetAsync(d_count, 0, sizeof(float));
			cudaStatus = cudaMemsetAsync(d_mse, 0, sizeof(float)*BatchSize);

			//=========================================================================================================================================	

			for (int j = PStart[p]; j < PStart[p] + partsize; j += BatchSize)
			{

				//=========================================================================================================================================	
				// At the start of each iteration the DataAugmentation() CUDA kernel is called to generate a batch after applying data augmentaion. For the 
				// training data-segements (p < NParts - VParts) DataAugmentation() is used, and for the validation data-segments (NParts - VParts) >= p >
				// NParts DataAugmentationValidate() is used.

				//=========================================================================================================================================	

				if (p < NParts - VParts)
				{
					Iter++;
					DataAugmentation <1> << <gridSize_Crop, blockSize1 >> >(d_X[0], Red + altr, Green + altr, Blue + altr, d_HeightTr + PStart[p], d_WidthTr + PStart[p], d_StartTr + PStart[p], d_Indx1 + j - PStart[p], (d_Crop + 3 * jAugm), (d_randRGB + 3 * jAugm), (d_Cropf + 2 * jAugm));
					jAugm += BatchSize;
				}
				else
				{
					DataAugmentationValidate <1> << <gridSize_Crop, blockSize1 >> >(d_X[0], Red + altr, Green + altr, Blue + altr, d_HeightTr + PStart[p], d_WidthTr + PStart[p], d_StartTr + PStart[p], d_Indx1 + j - PStart[p]);
				}

				//=========================================================================================================================================	
				//
				// Once the training batch was generated this batch will be passed forward through all CL convolutional layers. Therefore the next
				// for loop which will be repeated CL times will propagate the input signal froward through all CL convolutional layers. At each 
				// layer, the cudnnConvolutionForward() cudnn.lib function implements the forward pass of the convolution operation, and then the 
				// BatchNormForward1x() and BatchNormForward2() cuda kernels implements the forward pass of the BN operation. For layer k with 
				// PoolType[k] =1 the MaxPoolingForward() cuda kernel implements the forward pass of the max pooling operation. For layer k with 
				// PoolType[k] = 2 the GlobalAvgPoolingForward() cuda kernel implements the forward pass of the global average pooling operation.    
				// The variable JMP defines the number of convolutional layer that will be skipped by a residual connection. The condition 
				// (k > 0 && k%JMP == 0) means that layer k has an additional residual input coming from layer k-JMP. For such layers BatchNormForward22() 
				// will be used instead of BatchNormForward2() because the residual input is incorporated after BN and before ReLU activation, and both 
				// are combined in one cuda kernel to reduce GPU memory loads/stores.
				//
				//=========================================================================================================================================	

				float *d_ts, *d_t;

				for (int k = 0; k < CL; k++)
				{
					dnnStatus = cudnnConvolutionForward(dnnHandle, &alpha1, Desc_X[k], d_X[k], Desc_W[k], d_W[k], Conv_Desc[k], FwdAlg[k], d_ws, wsSize, &beta1, Desc_Y[k], d_Y[k]);

					BatchNormForward1a <BLOCKSIZE4> << <gridSizeBN1[k], blockSize4 >> >(d_SMU[k], d_Y[k], OutCh[k], Yr[k]);
					
					if (epoch == (NumEpoch - 1) && p < (NParts - VParts))
					{
						int j1 = (j - PStart[p]) / BatchSize;
						j1 = j1 % (NumTasks);
						
						//-----------------------------------------------------------------------------------------------------------------------------------------
						// The BatchNormForwardT1b() cuda kernel which will only be executed in the last epoch to calculate the fixed set of means and variance
						// that will be used in the inference stage, will calculate a separate set for each task. All these sets will be stored in the same GPU
						// buffer d_SMUs. The means and variances for all tasks for conv layer k are stored in d_SMUs[k].
						//-----------------------------------------------------------------------------------------------------------------------------------------

						BatchNormForwardT1b<BLOCKSIZE4> << <gridSizeBN11[k], blockSize4 >> >(d_SMU[k], d_SMUs[k] + j1 * 2 * OutCh[k], OutCh[k], Yr[k] * BatchSize);
					}
					else      BatchNormForward1b<BLOCKSIZE4> << <gridSizeBN11[k], blockSize4 >> >(d_SMU[k], OutCh[k], Yr[k] * BatchSize);

					//-----------------------------------------------------------------------------------------------------------------------------------------

					d_ts = (k < CL - 1) ? d_X[k + 1] : d_Y0;

					if (k > 0 && k%JMP == 0)
					{
						d_t = d_X[k - JMP + 1];
						if (OutCh[k - JMP] != OutCh[k])
						{
							dnnStatus = cudnnConvolutionForward(dnnHandle, &alpha1, Desc_Xs[k - JMP], d_X[k - JMP + 1], Desc_Ws[k - JMP], d_Ws[k - JMP], Conv_s_Desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, d_ws, wsSize, &beta1, Desc_Ys[k - JMP], d_YY);
							d_t = d_YY;
						}

						if (PoolType[k] == 0)
						{
							BatchNormForward22 <BLOCKSIZE4> << <gridSizeBN2[k], blockSize4 >> >(d_ts, d_Y[k], d_t, d_F[k], d_SMU[k], d_Param[k], OutCh[k], Yr[k]);
						}

						if (PoolType[k] == 1)
						{
							BatchNormForward22 <BLOCKSIZE4> << <gridSizeBN2[k], blockSize4 >> >(d_YY, d_Y[k], d_t, d_F[k], d_SMU[k], d_Param[k], OutCh[k], Yr[k]);
							MaxPoolingForward <1> << <gridSizeP[k], blockSize1 >> >(d_ts, d_YY, d_Indx[k], CR[k], SR[k], Sr1[k], Sr2[k], OutCh[k]);
						}

						if (PoolType[k] == 2)
						{
							BatchNormForward22 <BLOCKSIZE4> << <gridSizeBN2[k], blockSize4 >> >(d_YY, d_Y[k], d_t, d_F[k], d_SMU[k], d_Param[k], OutCh[k], Yr[k]);
							GlobalAvgPoolingForward <BLOCKSIZE3> << <gridSizePA, blockSize3 >> >(d_ts, d_YY, OutCh[k], CR[k] * CR[k]);
						}

					}
					else
					{
						if (PoolType[k] == 0)
						{
							BatchNormForward2 <BLOCKSIZE4> << <gridSizeBN2[k], blockSize4 >> >(d_ts, d_Y[k], d_SMU[k], d_Param[k], OutCh[k], Yr[k]);
						}

						if (PoolType[k] == 1)
						{
							BatchNormForward2 <BLOCKSIZE4> << <gridSizeBN2[k], blockSize4 >> >(d_YY, d_Y[k], d_SMU[k], d_Param[k], OutCh[k], Yr[k]);
							MaxPoolingForward <1> << <gridSizeP[k], blockSize1 >> >(d_ts, d_YY, d_Indx[k], CR[k], SR[k], Sr1[k], Sr2[k], OutCh[k]);
						}

						if (PoolType[k] == 2)
						{
							BatchNormForward2 <BLOCKSIZE4> << <gridSizeBN2[k], blockSize4 >> >(d_YY, d_Y[k], d_SMU[k], d_Param[k], OutCh[k], Yr[k]);
							GlobalAvgPoolingForward <BLOCKSIZE3> << <gridSizePA, blockSize3 >> >(d_ts, d_YY, OutCh[k], CR[k] * CR[k]);
						}
					}
				}

				//=========================================================================================================================================
				// the cublasSgemm() cuda matrix multiplication function implements the forward pass of the FC output layer. The SoftmaxForward() 
				// cuda kernel applies softmax activation to the outputs of the cublasSgemm() function and returns the error signal propagated back 
				// through the softmax stage in d_YF.
				//=========================================================================================================================================

				blasstatus = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Out1, BatchSize, In1 + 1, &alpha1, d_WF, Out1, d_Y0, In1 + 1, &beta1, d_YF, Out1);

				Softmax  <NumClasses, BLOCKSIZE2> << <BatchSize, BLOCKSIZE2 >> >(d_YF, d_T + PStart[p], d_Indx1 + j - PStart[p], d_mse, d_count);

				//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				//                                                       Back-Propagation                                                                || 
				//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




				//=========================================================================================================================================
				// NParts is the total number of data-segments which include the training and validation data-segments, and VParts is the number of 
				// validation data-segments only. p is the index of the current data-segment. The backward pass should only be executed for the training 
				// data-segments, and not for the validation data-segments, and therefore should only be executed for p < (NParts - VParts). The last 
				// epoch in this implementation is dedicated to calculate the fixed means and variances of BN that will be used in the inference stage,
				// which only requires the forward pass, and therefore the backward pass should only be executed for epoch < (Epoch - 1).
				//=========================================================================================================================================

				if (p < (NParts - VParts) && epoch < (NumEpoch - 1))
				{

					//=========================================================================================================================================
					//
					// The first stage in the back-propagation half is propagating the error signal back through the FC output layer. The first cublasSgemm() 
					// function calculates the derivatives of the weights of the FC output layer in d_DWF. The second cublasSgemm() function propagates back 
					// the error signal from the outputs of the output layer d_YF to the inputs of the output layer d_Y0.
					//
					//=========================================================================================================================================

					blasstatus = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, Out1, In1 + 1, BatchSize, &alpha1, d_YF, Out1, d_Y0, In1 + 1, &beta1, d_DWF, Out1);

					blasstatus = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, In1, BatchSize, Out1, &alpha1, d_WF, Out1, d_YF, Out1, &beta1, d_Y0, In1 + 1);

					//=========================================================================================================================================
					//
					// Then the error signal is propagated back through all CL convolutional layers to update all trainable parameters in those layers. 
					// d_DParam[k] and d_Derv[k] are updated by accumulating different thread block results using atomicadds() and therefore they 
					// need to be reset to zero. The error signal propagates back through a convolutional layer in a reverse order, through the pooling
					// stage (if there is one), then through the BN stage, and finally through the convolution stage. 
					// For conv layer k where (k%JMP == 0 && k + JMP < CL) two backpropagated signals will be propagated back through to that layer's 
					// output, one from conv layer k+1 and one from the residual connection that connects this output forward to layer k+JMP.
					// For conv layer = where (k > 0 && k%JMP == 0) the BatchNormBackward22() BN kernel will be used instead of BatchNormBackward2(),
					// because that layer has an additional residual input coming from layer k-JMP.
					// The BatchNormBackward2() and BatchNormBackward1() kernels implement the backward pass for BN.
					// For (PoolType[k] == 1) the MaxPoolingBackward() kernel implements the backward pass for the maxpooling stage for conv layer k.
					// For (PoolType[k] == 2) the GlobalAvgPoolingBackward() kernel implements the backward pass for the average pooling stage for conv layer k.
					// The cudnnConvolutionBackwardFilter(k) cudnn.lib function calculates the derivatives for the weights of conv layer k.
					// The cudnnConvolutionBackwardData(k) cudnn.lib function propagates the error signal back from the output channels side to the input
					// channels side for conv layer k.
					//
					//=========================================================================================================================================

					for (int k = CL - 1; k >= 0; k--)
					{
						cudaStatus = cudaMemsetAsync(d_DParam[k], 0, sizeof(float) * 2 * OutCh[k]);
						cudaStatus = cudaMemsetAsync(d_Derv[k], 0, sizeof(float) * 2 * OutCh[k]);

						//-----------------------------------------------------------------------------------------------------------------------------------------					

						d_ts = (k < CL - 1) ? d_X[k + 1] : d_Y0;

						if (k%JMP == 0 && k + JMP < CL)
						{
							d_t = (PoolType[k + JMP] == 0) ? d_X[k + JMP + 1] : d_YY;
							if (OutCh[k] == OutCh[k + JMP])   Add_Mtx <1> << <gridSizeAddYB[k], BLOCKSIZE1 >> >(d_ts, d_t, Xr[k + 1] * Xc[k + 1]);
							else							  dnnStatus = cudnnConvolutionBackwardData(dnnHandle, &alpha2, Desc_Ws[k], d_Ws[k], Desc_Ys[k], d_t, Conv_s_Desc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, d_ws, wsSize, &beta2, Desc_Xs[k], d_ts);
						}

						//-----------------------------------------------------------------------------------------------------------------------------------------

						if (k > 0 && k%JMP == 0)
						{
							if (PoolType[k] == 0)
							{
								BatchNormBackward22 <BLOCKSIZE4> << <gridSizeBN1[k], blockSize4 >> >(d_DParam[k], d_Derv[k], d_Param[k], d_SMU[k], d_ts, d_F[k], d_Y[k], OutCh[k], Yr[k]);
								BatchNormBackward1 <BLOCKSIZE4> << <gridSizeBN2[k], blockSize4 >> >(d_Y[k], d_ts, d_Param[k], d_SMU[k], d_Derv[k], OutCh[k], Yr[k]);
							}

							if (PoolType[k] == 1)
							{
								cudaStatus = cudaMemsetAsync(d_YY, 0, Yr[k] * Yc[k] * sizeof(float));
								MaxPoolingBackward <1> << <gridSizeP[k], blockSize1 >> >(d_YY, d_ts, d_Indx[k], SR[k] * SR[k], OutCh[k]);
								BatchNormBackward22 <BLOCKSIZE4> << <gridSizeBN1[k], blockSize4 >> >(d_DParam[k], d_Derv[k], d_Param[k], d_SMU[k], d_YY, d_F[k], d_Y[k], OutCh[k], Yr[k]);
								BatchNormBackward1 <BLOCKSIZE4> << <gridSizeBN2[k], blockSize4 >> >(d_Y[k], d_YY, d_Param[k], d_SMU[k], d_Derv[k], OutCh[k], Yr[k]);
							}

							if (PoolType[k] == 2)
							{
								GlobalAvgPoolingBackward <BLOCKSIZE3> << <gridSizePA, blockSize3 >> >(d_YY, d_ts, OutCh[k], CR[k] * CR[k]);
								BatchNormBackward22 <BLOCKSIZE4> << <gridSizeBN1[k], blockSize4 >> >(d_DParam[k], d_Derv[k], d_Param[k], d_SMU[k], d_YY, d_F[k], d_Y[k], OutCh[k], Yr[k]);
								BatchNormBackward1 <BLOCKSIZE4> << <gridSizeBN2[k], blockSize4 >> >(d_Y[k], d_YY, d_Param[k], d_SMU[k], d_Derv[k], OutCh[k], Yr[k]);
							}
						}
						else
						{
							if (OutCh[k - 1] != OutCh[k + JMP - 1] && (k + JMP - 1) % JMP == 0)
							{
								d_t = (PoolType[k + JMP - 1] == 0) ? d_X[k + JMP] : d_YY;
								dnnStatus = cudnnConvolutionBackwardFilter(dnnHandle, &alpha1, Desc_Xs[k - 1], d_X[k], Desc_Ys[k - 1], d_t, Conv_s_Desc, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, d_ws, wsSize, &beta1, Desc_Ws[k - 1], d_DWs[k - 1]);
							}

							if (PoolType[k] == 0)
							{
								BatchNormBackward2 <BLOCKSIZE4> << <gridSizeBN1[k], blockSize4 >> >(d_DParam[k], d_Derv[k], d_Param[k], d_SMU[k], d_ts, d_Y[k], OutCh[k], Yr[k]);
								BatchNormBackward1 <BLOCKSIZE4> << <gridSizeBN2[k], blockSize4 >> >(d_Y[k], d_ts, d_Param[k], d_SMU[k], d_Derv[k], OutCh[k], Yr[k]);
							}

							if (PoolType[k] == 1)
							{

								cudaStatus = cudaMemsetAsync(d_YY, 0, Yr[k] * Yc[k] * sizeof(float));
								MaxPoolingBackward <1> << <gridSizeP[k], blockSize1 >> >(d_YY, d_ts, d_Indx[k], SR[k] * SR[k], OutCh[k]);
								BatchNormBackward2 <BLOCKSIZE4> << <gridSizeBN1[k], blockSize4 >> >(d_DParam[k], d_Derv[k], d_Param[k], d_SMU[k], d_YY, d_Y[k], OutCh[k], Yr[k]);
								BatchNormBackward1 <BLOCKSIZE4> << <gridSizeBN2[k], blockSize4 >> >(d_Y[k], d_YY, d_Param[k], d_SMU[k], d_Derv[k], OutCh[k], Yr[k]);
							}

							if (PoolType[k] == 2)
							{
								GlobalAvgPoolingBackward <BLOCKSIZE3> << <gridSizePA, blockSize3 >> >(d_YY, d_ts, OutCh[k], CR[k] * CR[k]);
								BatchNormBackward2 <BLOCKSIZE4> << <gridSizeBN1[k], blockSize4 >> >(d_DParam[k], d_Derv[k], d_Param[k], d_SMU[k], d_YY, d_Y[k], OutCh[k], Yr[k]);
								BatchNormBackward1 <BLOCKSIZE4> << <gridSizeBN2[k], blockSize4 >> >(d_Y[k], d_YY, d_Param[k], d_SMU[k], d_Derv[k], OutCh[k], Yr[k]);
							}
						}

						//-----------------------------------------------------------------------------------------------------------------------------------------

						dnnStatus = cudnnConvolutionBackwardFilter(dnnHandle, &alpha1, Desc_X[k], d_X[k], Desc_Y[k], d_Y[k], Conv_Desc[k], BwdFilterAlg[k], d_ws, wsSize, &beta1, Desc_W[k], d_DW[k]);

						//-----------------------------------------------------------------------------------------------------------------------------------------

						if (k > 0) dnnStatus = cudnnConvolutionBackwardData(dnnHandle, &alpha1, Desc_W[k], d_W[k], Desc_Y[k], d_Y[k], Conv_Desc[k], BwdDataAlg[k], d_ws, wsSize, &beta1, Desc_X[k], d_X[k]);

						//-----------------------------------------------------------------------------------------------------------------------------------------

					}

					//=========================================================================================================================================
					// Once all the derivatives are ready the  Update_RMSprop1() cuda kernel is invoked to update the weights of the conv layers, and the 
					// weights of the residual connections, and the weights of the FC output layer. The Update_RMSprop2() is called to update the trainable
					// parameters of the BN stages. Update_RMSprop1() and Update_RMSprop2() use RMSprop version of SGD to update the network parameters.  
					//=========================================================================================================================================

					for (int k = 0; k < CL; k++)
					{
						Update_RMSprop1<1> << <gridSizeAddW[k], BLOCKSIZE1 >> >(d_W[k], d_V[k], d_DW[k], lr, lmda, WSize[k], Iter);
						Update_RMSprop2<1> << <gridSizeAddA[k], blockSize1 >> >(d_Param[k], d_ParamV[k], d_DParam[k], lr, lmda, 2 * OutCh[k], Iter);
					}

					for (int k = 0; k < CL - JMP; k += JMP)
					{
						if (OutCh[k] != OutCh[k + JMP])
							Update_RMSprop1<1> << <gridSizeAddWs[k], BLOCKSIZE1 >> >(d_Ws[k], d_Vs[k], d_DWs[k], lr, lmda, WsSize[k], Iter);
					}

					Update_RMSprop1<1> << <gridSizeAddWF, BLOCKSIZE1 >> >(d_WF, d_VF, d_DWF, lr, lmda, Out1*(In1 + 1), Iter);

				}

			}

			//=========================================================================================================================================
			// PrintIterResults() prints the mse and classification rates for the training and validation sets per epoch. Then the learning rate 
			// is decayed at specific intervals (specific epochs) defined in array DecAlpha[]. SaveParameters2() saves the network parameters 
			// including the running averages required by RMSprop once in every 10 epochs in case training was interrupted the network can resume
			// training from the last saved set of parameters. 
			//=========================================================================================================================================

			PrintIterResults(outResults, d_mse, d_count, PStart, NParts, VParts, i, TrainSizeM, ValSizeM);

			if (p == (NParts - 1) && (epoch + 1) == DecAlpha[kk]) { lr = lr*0.4f; kk++; }

			if ((epoch + 1) % 50 == 0) SaveParameters2(outParamCopy, TEMP, d_W, d_V, d_Ws, d_Vs, d_WF, d_VF, d_Param, d_ParamV, WSize, WsSize, In1, i + 1, Iter, kk, lr);

			//=========================================================================================================================================
			// One the current training epoch is finished the main CPU thread synchronizes with the other thread that reads the data from the desk
			// to the main memory (producer-consumer synchronization). This guarantees that there is data available to be consumed by the main thread 
			// that runs the CNN.
			//=========================================================================================================================================

			unique_lock<mutex> locker(mu);

			slot--;
			while (slot == 0 && i < (NParts*NumEpoch - 1) && MSE < 100)
				not_empty.wait(locker);

			not_full.notify_one();
			locker.unlock();

		}

		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		//
		//   End of main Forloop, end of training phase
		//
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		t1.join();

		SaveParameters1(outParam, TEMP, d_W, d_Ws, d_WF, d_Param, WSize, WsSize, In1);
		fclose(outParam);
		fclose(outParamCopy);



		//=========================================================================================================================================
		// AdjustFixedMeansStds() averages the accumulated means and variances in d_SMUs to be used as fixed sets in the inference stage. However,
		// there are NumTasks sets of fixed means and variance, one for each task.
		//=========================================================================================================================================

		for (int i = 0; i < CL; i++)
		{
			dim3 gridSizeAddAs((2 * OutCh[i] * NumTasks + BLOCKSIZE1 - 1) / BLOCKSIZE1, 1, 1);
			AdjustFixedMeansStds <1> << <gridSizeAddAs, blockSize1 >> >(d_SMUs[i], 2 * OutCh[i] * NumTasks, TrainSizeM);
			cudaStatus = cudaMemcpy(TEMP, d_SMUs[i], sizeof(float) * 2 * OutCh[i] * NumTasks, cudaMemcpyDeviceToHost);
			fwrite(TEMP, sizeof(float), 2 * OutCh[i] * NumTasks, outMeansVariances);
		}
		fclose(outMeansVariances);
	}




	//=========================================================================================================================================
	// cudaEventElapsedTime() runtime cuda function returns the total time in milliseconds spent in the training phase. SaveParameters() saves 
	// the network parameters. FreeTrainSpecificData() frees all memory that is required in the training stage but it is not required in the 
	// inference stage.
	//=========================================================================================================================================

	cudaStatus = cudaEventRecord(stop, 0);
	cudaStatus = cudaEventSynchronize(stop);
	float time1;
	cudaStatus = cudaEventElapsedTime(&time1, start, stop);

	cout << endl << "time = " << time1 << endl << endl;
	fprintf(outResults, "\n\ntime = %f\n\n ", time1);

	FreeTrainSpecificData(d_V, d_DW, d_Vs, d_DWs, d_DParam, d_ParamV, d_Derv, d_VF, d_DWF, d_count, d_rand1, d_randRGB, d_Cropf, d_Indx1, Indx, Indx1, d_Crop, NParts);


	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	//
	//                                                       Inference phase.
	//
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	if (Operation_Mode == INFERENCE || Operation_Mode == TRAIN_PLUS_INFERENCE || Operation_Mode == INTERRUPTED_TRAIN_PLUS_INFERENCE)
	{
		//======================================================================================================================================================
		// In the case of doing inference only, a previously stored copy of the fixed sets of means and variances (produced by the same network used for inference) 
		// are reloaded into d_SMUs (using the inMeansVariances file pointer). There will be NumTasks sets of fixed means and variances, one for each task.
		//======================================================================================================================================================

		if (Operation_Mode == INFERENCE)
		{
			for (int i = 0; i < CL; i++)
			{
				size_t numread = fread(TEMP, sizeof(float), 2 * OutCh[i] * NumTasks, inMeansVariances);
				cudaStatus = cudaMemcpy(d_SMUs[i], TEMP, sizeof(float) * 2 * OutCh[i] * NumTasks, cudaMemcpyHostToDevice);
			}
			fclose(inMeansVariances);
		}

		int *d_MTX;

		//======================================================================================================================================================
		// The InitializeTrainingData() function initializes the d_HeightTr, d_WidthTr, and d_StartTr GPU buffers with the heights, widths, and starting 
		// memory addresses of the test images. InitializeMultiCropInference() allocates and initializes the d_MTX GPU buffer with the cropping positions and 
		// and scales that will be used with multi-crop Inference. The three input files in6, in7, and in8 are opened to read the RGB input channels of the 
		// test images.
		//======================================================================================================================================================

		InitializeTrainingData(&d_HeightTr, &d_WidthTr, &d_StartTr, &d_T, &PStart, &PartSize, &StartPart, &NParts, &VParts, TestSize);
		InitializeMultiCropInference(&d_MTX);
		FILE *in6, *in7, *in8;

		char FileName[128];

		strcpy(FileName, DataFloder); strcat(FileName, "valRed.txt");
		in6 = fopen(FileName, "rb");
		strcpy(FileName, DataFloder); strcat(FileName, "valGreen.txt");
		in7 = fopen(FileName, "rb");
		strcpy(FileName, DataFloder); strcat(FileName, "valBlue.txt");
		in8 = fopen(FileName, "rb");

		unsigned int *d_flip;
		DividTestsetIntoMultipleTasks(&d_flip, &Indx1, Indx, CountImgeTask, d_T, PStart, NParts, cuda_gen);

		//=====================================================================================================================================================================	
		// Alocate d_Yss GPU buffer that will be used to store the whole output labels predicted by the network. Alocate d_flip GPU buffer that will be initialized by 
		// random integers used to decide on horizontal flipping.
		//=====================================================================================================================================================================	

		float  *d_Yss;
		cudaStatus = cudaMalloc(&d_Yss, sizeof(float) * NumClasses*TestSize);
		cudaStatus = cudaMemset(d_mse, 0, sizeof(float)*BatchSize);
		cudaStatus = cudaMemset(d_Yss, 0, sizeof(float)*TestSize*NumClasses);

		//unsigned int *d_flip;
		//cudaStatus = cudaMalloc(&d_flip, sizeof(int)*TestSize*EpochTs);




		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		//
		//                                                      Start of the inference Forloop.
		//
		// This for loop is similar to the one used by the single-task network, the main difference is using multiple sets of fixed means and variances 
		// instead of a single set for a single task. Also the images need to be shuffled and arranged so that each task can use its own set of means 
		// and variances, and therefore the DataAugmentationInference2() and SoftmaxInference2() cuda kernels are slightly different form the one 
		// used for the single task implementation which doesn't require rearranging  the test images.
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		jAugm = 0;

		for (int i = 0; i < NParts; i++)
		{
			int p = i%NParts;

			//=========================================================================================================================================	
			// curandGenerate() initializes d_flip with random integer numbers. The fread() function reads one data segment of the test set input RGB 
			// channels into the Red, Green, and Blue buffers.
			//=========================================================================================================================================	

			curandStatus = curandGenerate(cuda_gen, d_flip, (PStart[p + 1] - PStart[p])*EpochTs);

			size_t numread10 = fread(Red, sizeof(unsigned char), PartSize[p], in6);
			size_t numread11 = fread(Green, sizeof(unsigned char), PartSize[p], in7);
			size_t numread12 = fread(Blue, sizeof(unsigned char), PartSize[p], in8);

			//=========================================================================================================================================	
			// Again the ReshuffleImagesMT() function will arrange the images of different tasks in different batches, which allows each task to use
			// its own fixed set of BN means and variances. Because this function is called once for each data segment, it is not reshuffling the
			// images as is the case in the training phase where this function is called in each training epoch.
			//=========================================================================================================================================	

			int partsize;
			ReshuffleImagesMT(&partsize, Indx1, Indx, CountImgeTask, PStart, p);
			cudaStatus = cudaMemcpyAsync(d_Indx1, Indx1, sizeof(int) * (PStart[p + 1] - PStart[p]), cudaMemcpyHostToDevice);

			//=========================================================================================================================================	
			// Once the test data is ready in the memory, the test images are passed forward to calculate the test image labels in d_Yss
			//=========================================================================================================================================	
			
			for (int j = PStart[p]; j < PStart[p + 1]; j += BatchSize)
			{
				if (j % 500 == 0)cout << j << "   ";

				int j1 = (j - PStart[p]) / BatchSize;
				j1 = j1 % NumTasks;

				for (int epoch = 0; epoch < EpochTs; epoch++)
				{										
					DataAugmentationInference2 <EpochTs> << <gridSize_Crop, blockSize1 >> >(d_X[0], Red, Green, Blue, d_HeightTr + PStart[p], d_WidthTr + PStart[p], d_StartTr + PStart[p], d_Indx1 + j - PStart[p], d_MTX + 3 * EpochTs*PStart[p], d_flip + jAugm, epoch);
					jAugm += BatchSize;

					//=========================================================================================================================================	

					float *d_ts, *d_t;

					for (int k = 0; k < CL; k++)
					{
						//-----------------------------------------------------------------------------------------------------------------------------------------	

						dnnStatus = cudnnConvolutionForward(dnnHandle, &alpha1, Desc_X[k], d_X[k], Desc_W[k], d_W[k], Conv_Desc[k], FwdAlg[k], d_ws, wsSize, &beta1, Desc_Y[k], d_Y[k]);

						//-----------------------------------------------------------------------------------------------------------------------------------------	
						// The BatchNormForward2() and BatchNormForward22() cuda kernels that implement the second stage of the forward pass of BN and using the 
						// input buffer d_SMUs[k] + j1 * 2 * OutCh[k] to access the fixed set of means and variances for task j1.
						//-----------------------------------------------------------------------------------------------------------------------------------------	

						d_ts = (k < CL - 1) ? d_X[k + 1] : d_Y0;


						if (k > 0 && k%JMP == 0)
						{
							d_t = d_X[k - JMP + 1];
							if (OutCh[k - JMP] != OutCh[k])
							{
								dnnStatus = cudnnConvolutionForward(dnnHandle, &alpha1, Desc_Xs[k - JMP], d_X[k - JMP + 1], Desc_Ws[k - JMP], d_Ws[k - JMP], Conv_s_Desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, d_ws, wsSize, &beta1, Desc_Ys[k - JMP], d_YY);
								d_t = d_YY;
							}

							if (PoolType[k] == 0)
							{
								BatchNormForward22 <BLOCKSIZE4> << <gridSizeBN2[k], blockSize4 >> >(d_ts, d_Y[k], d_t, d_F[k], d_SMUs[k] + j1 * 2 * OutCh[k], d_Param[k], OutCh[k], Yr[k]);
							}

							if (PoolType[k] == 1)
							{
								BatchNormForward22 <BLOCKSIZE4> << <gridSizeBN2[k], blockSize4 >> >(d_YY, d_Y[k], d_t, d_F[k], d_SMUs[k] + j1 * 2 * OutCh[k], d_Param[k], OutCh[k], Yr[k]);
								MaxPoolingForward <1> << <gridSizeP[k], blockSize1 >> >(d_ts, d_YY, d_Indx[k], CR[k], SR[k], Sr1[k], Sr2[k], OutCh[k]);
							}

							if (PoolType[k] == 2)
							{
								BatchNormForward22 <BLOCKSIZE4> << <gridSizeBN2[k], blockSize4 >> >(d_YY, d_Y[k], d_t, d_F[k], d_SMUs[k] + j1 * 2 * OutCh[k], d_Param[k], OutCh[k], Yr[k]);
								GlobalAvgPoolingForward<BLOCKSIZE3> << <gridSizePA, blockSize3 >> >(d_ts, d_YY, OutCh[k], CR[k] * CR[k]);
							}

						}
						else
						{
							if (PoolType[k] == 0)
							{
								BatchNormForward2 <BLOCKSIZE4> << <gridSizeBN2[k], blockSize4 >> >(d_ts, d_Y[k], d_SMUs[k] + j1 * 2 * OutCh[k], d_Param[k], OutCh[k], Yr[k]);
							}

							if (PoolType[k] == 1)
							{
								BatchNormForward2 <BLOCKSIZE4> << <gridSizeBN2[k], blockSize4 >> >(d_YY, d_Y[k], d_SMUs[k] + j1 * 2 * OutCh[k], d_Param[k], OutCh[k], Yr[k]);
								MaxPoolingForward<1> << <gridSizeP[k], blockSize1 >> >(d_ts, d_YY, d_Indx[k], CR[k], SR[k], Sr1[k], Sr2[k], OutCh[k]);
							}

							if (PoolType[k] == 2)
							{
								BatchNormForward2 <BLOCKSIZE4> << <gridSizeBN2[k], blockSize4 >> >(d_YY, d_Y[k], d_SMUs[k] + j1 * 2 * OutCh[k], d_Param[k], OutCh[k], Yr[k]);
								GlobalAvgPoolingForward<BLOCKSIZE3> << <gridSizePA, blockSize3 >> >(d_ts, d_YY, OutCh[k], CR[k] * CR[k]);
							}
						}
					}

					//=========================================================================================================================================


					blasstatus = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Out1, BatchSize, In1 + 1, &alpha1, d_WF, Out1, d_Y0, In1 + 1, &beta1, d_YF, Out1);

					//-----------------------------------------------------------------------------------------------------------------------------------------

					SoftmaxInference2 <NumClasses, BLOCKSIZE2> << <BatchSize, BLOCKSIZE2 >> >(d_Yss + PStart[p] * NumClasses, d_YF, d_T + PStart[p], d_Indx1 + j - PStart[p], d_mse);

					//=========================================================================================================================================

				}
			}
		}

		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		//
		//                                                              End of the inference Forloop. 
		//
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		//-----------------------------------------------------------------------------------------------------------------------------------------
		// The PrintFinalResults() function prints the recognition rate for all tasks, in addition to the confusion matrix for the whole main data
		// set.
		//-----------------------------------------------------------------------------------------------------------------------------------------

		PrintFinalResults(outResults, d_Yss, d_mse, d_T);

		cudaFree(d_MTX); cudaFree(d_Yss); cudaFree(d_flip);
	}

	FreeRemainingMem(Red, Green, Blue, d_W, d_Ws, d_X, d_Y, d_Param, d_SMU, d_SMUs, d_Indx, d_F, d_WF, d_YF, d_Yv, d_YY, d_Y0, d_ws, d_mse, d_HeightTr, d_WidthTr, d_StartTr, d_T, PStart, StartPart, PartSize, TEMP);
	fclose(outResults);

	cudaDeviceReset();

	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	//
	//                                                                          End of program
	//
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

}
