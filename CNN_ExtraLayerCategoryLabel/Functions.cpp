#include"Header.h"

//================================================================================================================================================================
//================================================================================================================================================================
//================================================================================================================================================================
//  This .cpp file contains the implementation of functions required to initialize the network, save and reload the network paramters, and print out the results.
//================================================================================================================================================================
//================================================================================================================================================================
//================================================================================================================================================================



void SaveParameters1(FILE *out, float *TEMP, float **d_W, float **d_Ws, float *d_WF, float **d_Param, int *WSize, int *WsSize)
{

	//----------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This function saves the network parameters after training finishs.
	*/

	/****Argument List****/
	/*

	out:-      output .txt file.
	TEMP:-     temporary buffer to copy data from GPU memory to main memory.
	d_W:-      weight matrix that contains the weights of all convolutional layers.
	d_Ws:-     weight matrix that contains the weights of the residual connections.
	d_WF:-     weight matrix that contains the weights of the output FC layer.
	d_Param:-  matrix that conatins the trainable parameters of BN for all hidden layers.
	WSize:-    array contains the sizes of the weight matrcies of all convolutional layers.
	WsSize:-   array contains the sizes of the weight matrcies of all residual connections.
	In1:-      the input size of the output layer.

	*/
	//----------------------------------------------------------------------------------------------------------------------------------------------

	cudaError_t cudaStatus;

	for (int i = 0; i < CL - JMP; i += JMP)
	{
		if (OutCh[i] != OutCh[i + JMP])
		{
			cudaStatus = cudaMemcpy(TEMP, d_Ws[i], sizeof(float)*WsSize[i], cudaMemcpyDeviceToHost);
			fwrite(TEMP, sizeof(float), WsSize[i], out);
		}
	}

	for (int i = 0; i < CL; i++)
	{
		cudaStatus = cudaMemcpy(TEMP, d_W[i], sizeof(float)*WSize[i], cudaMemcpyDeviceToHost);
		fwrite(TEMP, sizeof(float), WSize[i], out);
	}

	cudaStatus = cudaMemcpy(TEMP, d_WF, sizeof(float) *Out1*(In1 + 1), cudaMemcpyDeviceToHost);
	fwrite(TEMP, sizeof(float), Out1*(In1 + 1), out);

	for (int i = 0; i < CL; i++)
	{
		cudaStatus = cudaMemcpy(TEMP, d_Param[i], sizeof(float) * 2 * OutCh[i], cudaMemcpyDeviceToHost);
		fwrite(TEMP, sizeof(float), 2 * OutCh[i], out);
	}
}

//============================================================================================================================================================

void ReloadParameters1(FILE *in, float *TEMP, float **d_W, float **d_Ws, float *d_WF, float **d_Param, int *WSize, int *WsSize)
{
	//----------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This function reloads the parameters of a previously trained network.
	*/

	/****Argument List****/
	/*

	out:-      output .txt file.
	TEMP:-     temporary buffer to copy data from GPU memory to main memory.
	d_W:-      weight matrix that contains the weights of all convolutional layers.
	d_Ws:-     weight matrix that contains the weights of the residual connections.
	d_WF:-     weight matrix that contains the weights of the output FC layer.
	d_Param:-  matrix that conatins the trainable parameters of BN for all hidden layers.
	WSize:-    array contains the sizes of the weight matrcies of all convolutional layers.
	WsSize:-   array contains the sizes of the weight matrcies of all residual connections.
	In1:-      the input size of the output layer.

	*/
	//----------------------------------------------------------------------------------------------------------------------------------------------

	cudaError_t cudaStatus;
	size_t numread;

	for (int i = 0; i < CL - JMP; i += JMP)
	{
		if (OutCh[i] != OutCh[i + JMP])
		{
			numread = fread(TEMP, sizeof(float), WsSize[i], in);
			cudaStatus = cudaMemcpy(d_Ws[i], TEMP, sizeof(float)*WsSize[i], cudaMemcpyHostToDevice);
		}
	}

	for (int i = 0; i < CL; i++)
	{
		numread = fread(TEMP, sizeof(float), WSize[i], in);
		cudaStatus = cudaMemcpy(d_W[i], TEMP, sizeof(float)*WSize[i], cudaMemcpyHostToDevice);
	}

	numread = fread(TEMP, sizeof(float), Out1*(In1 + 1), in);
	cudaStatus = cudaMemcpy(d_WF, TEMP, sizeof(float)*Out1*(In1 + 1), cudaMemcpyHostToDevice);

	for (int i = 0; i < CL; i++)
	{
		numread = fread(TEMP, sizeof(float), 2 * OutCh[i], in);
		cudaStatus = cudaMemcpy(d_Param[i], TEMP, sizeof(float) * 2 * OutCh[i], cudaMemcpyHostToDevice);
	}

}

//============================================================================================================================================================

void SaveParameters2_PlusExtraLayer(FILE *out, float *TEMP, float **d_W, float **d_V, float **d_Ws, float **d_Vs, float *d_WF, float *d_VF, float *d_WF2, float *d_VF2, float **d_Param, float **d_ParamV, int *WSize, int *WsSize, int iter0, int Iter0, int kk0, float lr0)
{
	//----------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This function saves the network parameters multiple times throughtout training in case something interrupts the training process. This
	allows training to resume from the last saved set of parameters.
	*/

	/****Argument List****/
	/*

	out:-      output .txt file.
	TEMP:-     temporary buffer to copy data from GPU memory to main memory.
	d_W:-      weight matrix that contains the weights of all convolutional layers.
	d_V:-      matrix contains the current running average of the squared derivative of the weights of all convolutional layers.
	d_Ws:-     weight matrix that contains the weights of the residual connections.
	d_Vs:-     matrix contains the current running average of the squared derivative of the weights of the residual connections.
	d_WF:-     weight matrix that contains the weights of the output FC layer.
	d_VF:-     matrix contains the current running average of the squared derivative of the weights of the output layer.
	d_WF2:-    weight matrix that contains the weights of the extra output FC layer.
	d_VF2:-    matrix contains the current running average of the squared derivative of the weights of the extra output layer.
	d_Param:-  matrix that conatins the trainable parameters of BN for all hidden layers.
	d_ParamV:- matrix contains the current running average of the squared derivative of the trainable parameters of BN.
	WSize:-    array contains the sizes of the weight matrcies of all convolutional layers.
	WsSize:-   array contains the sizes of the weight matrcies of all residual connections.
	In1:-      the input size of the output layer.
	epoch:-    current training epoch.
	Iter:-     current training iteration.

	*/
	//----------------------------------------------------------------------------------------------------------------------------------------------

	cudaError_t cudaStatus;

	fseek(out, long(0), SEEK_SET);

	for (int i = 0; i < CL - JMP; i += JMP)
	{
		if (OutCh[i] != OutCh[i + JMP])
		{
			cudaStatus = cudaMemcpy(TEMP, d_Ws[i], sizeof(float)*WsSize[i], cudaMemcpyDeviceToHost);
			fwrite(TEMP, sizeof(float), WsSize[i], out);
			cudaStatus = cudaMemcpy(TEMP, d_Vs[i], sizeof(float)*WsSize[i], cudaMemcpyDeviceToHost);
			fwrite(TEMP, sizeof(float), WsSize[i], out);
		}
	}

	for (int i = 0; i < CL; i++)
	{
		cudaStatus = cudaMemcpy(TEMP, d_W[i], sizeof(float)*WSize[i], cudaMemcpyDeviceToHost);
		fwrite(TEMP, sizeof(float), WSize[i], out);
		cudaStatus = cudaMemcpy(TEMP, d_V[i], sizeof(float)*WSize[i], cudaMemcpyDeviceToHost);
		fwrite(TEMP, sizeof(float), WSize[i], out);
	}

	cudaStatus = cudaMemcpy(TEMP, d_WF, sizeof(float) *Out1*(In1 + 1), cudaMemcpyDeviceToHost);
	fwrite(TEMP, sizeof(float), Out1*(In1 + 1), out);
	cudaStatus = cudaMemcpy(TEMP, d_VF, sizeof(float) *Out1*(In1 + 1), cudaMemcpyDeviceToHost);
	fwrite(TEMP, sizeof(float), Out1*(In1 + 1), out);

	cudaStatus = cudaMemcpy(TEMP, d_WF2, sizeof(float) *Out2*(In2 + 1), cudaMemcpyDeviceToHost);
	fwrite(TEMP, sizeof(float), Out2*(In2 + 1), out);
	cudaStatus = cudaMemcpy(TEMP, d_VF2, sizeof(float) *Out2*(In2 + 1), cudaMemcpyDeviceToHost);
	fwrite(TEMP, sizeof(float), Out2*(In2 + 1), out);

	for (int i = 0; i < CL; i++)
	{
		cudaStatus = cudaMemcpy(TEMP, d_Param[i], sizeof(float) * 2 * OutCh[i], cudaMemcpyDeviceToHost);
		fwrite(TEMP, sizeof(float), 2 * OutCh[i], out);
		cudaStatus = cudaMemcpy(TEMP, d_ParamV[i], sizeof(float) * 2 * OutCh[i], cudaMemcpyDeviceToHost);
		fwrite(TEMP, sizeof(float), 2 * OutCh[i], out);
	}

	fprintf(out, "%d\t%d\t%d\t%f", iter0, Iter0, kk0, lr0);
	fflush(out);
}

//============================================================================================================================================================

void ReloadParameters2_PlusExtraLayer(FILE *in, float *TEMP, float **d_W, float **d_V, float **d_Ws, float **d_Vs, float *d_WF, float *d_VF, float *d_WF2, float *d_VF2, float **d_Param, float **d_ParamV, int *WSize, int *WsSize, int *iter0, int *Iter0, int *kk0, float *lr0)
{
	//----------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This function reloads the network parameters including the running averages of the squared derivatives in case something has previously
	interrupted the training process. This allows training to resume from the last saved set of parameters.
	*/

	/****Argument List****/
	/*

	out:-      output .txt file.
	TEMP:-     temporary buffer to copy data from GPU memory to main memory.
	d_W:-      weight matrix that contains the weights of all convolutional layers.
	d_V:-      matrix contains the current running average of the squared derivative of the weights of all convolutional layers.
	d_Ws:-     weight matrix that contains the weights of the residual connections.
	d_Vs:-     matrix contains the current running average of the squared derivative of the weights of the residual connections.
	d_WF:-     weight matrix that contains the weights of the output FC layer.
	d_VF:-     matrix contains the current running average of the squared derivative of the weights of the output layer.
	d_WF2:-    weight matrix that contains the weights of the extra output FC layer.
	d_VF2:-    matrix contains the current running average of the squared derivative of the weights of the extra output layer.
	d_Param:-  matrix that conatins the trainable parameters of BN for all hidden layers.
	d_ParamV:- matrix contains the current running average of the squared derivative of the trainable parameters of BN.
	WSize:-    array contains the sizes of the weight matrcies of all convolutional layers.
	WsSize:-   array contains the sizes of the weight matrcies of all residual connections.
	In1:-      the input size of the output layer.
	epoch:-    current training epoch.
	Iter:-     current training iteration.

	*/
	//----------------------------------------------------------------------------------------------------------------------------------------------

	cudaError_t cudaStatus;
	size_t numread;

	for (int i = 0; i < CL - JMP; i += JMP)
	{
		if (OutCh[i] != OutCh[i + JMP])
		{
			numread = fread(TEMP, sizeof(float), WsSize[i], in);
			cudaStatus = cudaMemcpy(d_Ws[i], TEMP, sizeof(float)*WsSize[i], cudaMemcpyHostToDevice);
			numread = fread(TEMP, sizeof(float), WsSize[i], in);
			cudaStatus = cudaMemcpy(d_Vs[i], TEMP, sizeof(float)*WsSize[i], cudaMemcpyHostToDevice);
		}
	}

	for (int i = 0; i < CL; i++)
	{
		numread = fread(TEMP, sizeof(float), WSize[i], in);
		cudaStatus = cudaMemcpy(d_W[i], TEMP, sizeof(float)*WSize[i], cudaMemcpyHostToDevice);
		numread = fread(TEMP, sizeof(float), WSize[i], in);
		cudaStatus = cudaMemcpy(d_V[i], TEMP, sizeof(float)*WSize[i], cudaMemcpyHostToDevice);
	}

	numread = fread(TEMP, sizeof(float), Out1*(In1 + 1), in);
	cudaStatus = cudaMemcpy(d_WF, TEMP, sizeof(float)*Out1*(In1 + 1), cudaMemcpyHostToDevice);
	numread = fread(TEMP, sizeof(float), Out1*(In1 + 1), in);
	cudaStatus = cudaMemcpy(d_VF, TEMP, sizeof(float)*Out1*(In1 + 1), cudaMemcpyHostToDevice);

	numread = fread(TEMP, sizeof(float), Out2*(In2 + 1), in);
	cudaStatus = cudaMemcpy(d_WF2, TEMP, sizeof(float)*Out2*(In2 + 1), cudaMemcpyHostToDevice);
	numread = fread(TEMP, sizeof(float), Out2*(In2 + 1), in);
	cudaStatus = cudaMemcpy(d_VF2, TEMP, sizeof(float)*Out2*(In2 + 1), cudaMemcpyHostToDevice);

	for (int i = 0; i < CL; i++)
	{
		numread = fread(TEMP, sizeof(float), 2 * OutCh[i], in);
		cudaStatus = cudaMemcpy(d_Param[i], TEMP, sizeof(float) * 2 * OutCh[i], cudaMemcpyHostToDevice);
		numread = fread(TEMP, sizeof(float), 2 * OutCh[i], in);
		cudaStatus = cudaMemcpy(d_ParamV[i], TEMP, sizeof(float) * 2 * OutCh[i], cudaMemcpyHostToDevice);
	}

	fscanf(in, "%d\t%d\t%d\t%f", iter0, Iter0, kk0, lr0);
	
}

//============================================================================================================================================================

void InitializeTrainingData_PlusCatLabel(unsigned int **P_Height, unsigned int **P_Width, size_t **P_Start, int **P_T, int **P_T2, int **P_PStart, size_t **P_PartSize, size_t **P_StartPart, int *NParts, int *VParts, int Size)
{
	//----------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This function allocates and initializes the buffers that hold the heights, widths, and memory starting addresses of all training, validation,
	or test images. This function assumes that there are 3 input files each holding one of the RGB channels of all images. Also this function divids
	large data that exceeds the predefined size of (RGB_GPU_SIZE / 2) into segments to fit in the main memory.
	*/

	/****Argument List****/
	/*

	P_HeightTr:-    Pointer to the GPU buffer that will be allocated and initialized with the heights of the training\validation images.
	P_WidthTr:-     Pointer to the GPU buffer that will be allocated and initialized with the widths of the training\validation images.
	P_StartTr:-     Pointer to the GPU buffer that will be allocated and initialized with the starting positions the training\validation images.
	P_T :-          Pointer to the GPU buffer that will be allocated and initialized with the class image labels.
	P_T2 :-         Pointer to the GPU buffer that will be allocated and initialized with the coarse category image labels.
	P_PStart:-      pointer to a buffer that will be allocated and initialized with the startining image index of all data segments.
	P_PartSize:-    pointer to a buffer that will be allocated and initialized with the sizes of all data segments.
	P_StartPart:-   pointer to a buffer that will be allocated and initialized with the startining memory address of all data segments.
	NParts:-        number of data segments or parts.
	VParts :-        number of data segments of the validation set.

	*/
	//----------------------------------------------------------------------------------------------------------------------------------------------

	extern int Operation_Mode;
	FILE *in4, *in5, *in6, *in7, *in8, *in9, *in10;
	extern char DataFloder[];
	char FileName[128];
	cudaError_t cudaStatus;

	strcpy_s(FileName, 128, DataFloder); strcat_s(FileName, 128, "cordinatesTr.txt");
	fopen_s(&in4, FileName, "rb");
	strcpy_s(FileName, 128, DataFloder); strcat_s(FileName, 128, "trainlabels.txt");
	fopen_s(&in5, FileName, "rb");
	strcpy_s(FileName, 128, DataFloder); strcat_s(FileName, 128, "cordinatesVal.txt");
	fopen_s(&in9, FileName, "rb");
	strcpy_s(FileName, 128, DataFloder); strcat_s(FileName, 128, "vallabels.txt");
	fopen_s(&in10, FileName, "rb");
	strcpy_s(FileName, 128, DataFloder); strcat_s(FileName, 128, "valRed.txt");
	fopen_s(&in6, FileName, "rb");
	strcpy_s(FileName, 128, DataFloder); strcat_s(FileName, 128, "valGreen.txt");
	fopen_s(&in7, FileName, "rb");
	strcpy_s(FileName, 128, DataFloder); strcat_s(FileName, 128, "valBlue.txt");
	fopen_s(&in8, FileName, "rb");



	if (Operation_Mode == INFERENCE || Size == (TrainSizeM + ValSizeM))
	{
		cudaStatus = cudaMalloc(P_Height, sizeof(unsigned int)*Size);
		cudaStatus = cudaMalloc(P_Width, sizeof(unsigned int)*Size);
		cudaStatus = cudaMalloc(P_Start, sizeof(size_t)*Size);
		cudaStatus = cudaMalloc(P_T, sizeof(int) * Size);
		cudaStatus = cudaMalloc(P_T2, sizeof(int) * Size);

		*P_PStart = new int[300 + 1];
		*P_StartPart = new size_t[300 + 1];
		*P_PartSize = new size_t[300];
	}

	unsigned int *d_Height = *P_Height;
	unsigned int *d_Width = *P_Width;
	size_t *d_Start = *P_Start;
	int *d_T = *P_T, *d_T2 = *P_T2;

	int *PStart = *P_PStart;
	size_t *StartPart = *P_StartPart;
	size_t *PartSize = *P_PartSize;

	unsigned int *Height = new unsigned int[Size];
	unsigned int *Width = new unsigned int[Size];
	size_t *Start = new size_t[Size + 1];
	int *T = new int[Size];
	int *T2 = new int[Size];

	if (Size == TestSizeM)
	{
		size_t numread1 = fread(Height, sizeof(unsigned int), TestSizeM, in9);
		size_t numread2 = fread(Width, sizeof(unsigned int), TestSizeM, in9);
		size_t numread3 = fread(T, sizeof(int), TestSizeM, in10);

	}
	else
	{
		size_t numread1 = fread(Height, sizeof(unsigned int), TrainSize, in4);
		size_t numread2 = fread(Width, sizeof(unsigned int), TrainSize, in4);
		size_t numread3 = fread(T, sizeof(int), TrainSize, in5);
		size_t numread4 = fread(T2, sizeof(int), TrainSize, in5);

		numread1 = fread(Height + TrainSizeM, sizeof(unsigned int), ValSize, in9);
		numread2 = fread(Width + TrainSizeM, sizeof(unsigned int), ValSize, in9);
		numread3 = fread(T + TrainSizeM, sizeof(int), ValSize, in10);
		numread4 = fread(T2 + TrainSizeM, sizeof(int), ValSize, in10);
	}

	cudaStatus = cudaMemcpy(d_Height, Height, sizeof(unsigned int) * Size, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(d_Width, Width, sizeof(unsigned int) * Size, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(d_T, T, sizeof(int) * Size, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(d_T2, T2, sizeof(int) * Size, cudaMemcpyHostToDevice);
	

	//-----------------------------------------------------------------------------------------------

	Start[0] = 0;
	PStart[0] = 0;
	StartPart[0] = Start[0];
	*VParts = 0;
	int jj = 0;

	for (int i = 1; i <= Size; i++)
	{
		Start[i] = Start[i - 1] + Height[i - 1] * Width[i - 1];

		if (Start[i] - Start[PStart[jj]] > RGB_GPU_SIZE / 2)
		{
			jj++;
			PStart[jj] = ((i - 1) / BatchSize) * BatchSize;
			PartSize[jj - 1] = Start[PStart[jj]] - Start[PStart[jj - 1]];
			StartPart[jj] = Start[PStart[jj]];

			if (Size == (TrainSizeM + ValSizeM) && i > TrainSizeM) (*VParts)++;
		}
		else
		{
			if (Size == TestSizeM)
			{
				if (i == TestSizeM)
				{
					jj++;
					PStart[jj] = (i / BatchSize) * BatchSize;
					PartSize[jj - 1] = Start[PStart[jj]] - Start[PStart[jj - 1]];
					StartPart[jj] = Start[PStart[jj]];
				}

			}
			else
			{
				if (i == TrainSizeM || i == TrainSizeM + ValSizeM)
				{
					jj++;
					PStart[jj] = (i / BatchSize) * BatchSize;
					PartSize[jj - 1] = Start[PStart[jj]] - Start[PStart[jj - 1]];
					StartPart[jj] = Start[PStart[jj]];
					
					if (i == TrainSizeM + ValSizeM) (*VParts)++;
				}

			}
		}
	}

	*NParts = jj;

	//---------------------------------------------------------------------------------------------------

	for (int i = 0; i < *NParts; i++)
	{
		size_t k = Start[PStart[i]];
		for (int j = PStart[i]; j < PStart[i + 1]; j++)
		{
			Start[j] -= k;
		}
	}

	cudaStatus = cudaMemcpy(d_Start, Start, sizeof(size_t) * Size, cudaMemcpyHostToDevice);

	delete[] Height;
	delete[] Width;
	delete[] Start;
	delete[]T;
	fclose(in4); fclose(in5); fclose(in6); fclose(in7); fclose(in8); fclose(in9); fclose(in10);
}

//============================================================================================================================================================

void InitialzeCuDNN(cudnnTensorDescriptor_t *Desc_X, cudnnTensorDescriptor_t *Desc_Y, cudnnFilterDescriptor_t *Desc_W, cudnnConvolutionDescriptor_t *Conv_Desc, cudnnTensorDescriptor_t *Desc_Xs, cudnnTensorDescriptor_t *Desc_Ys, cudnnFilterDescriptor_t *Desc_Ws, cudnnConvolutionDescriptor_t *Conv_s_Desc, cudnnConvolutionFwdAlgo_t *FwdAlg, cudnnConvolutionBwdDataAlgo_t *BwdDataAlg, cudnnConvolutionBwdFilterAlgo_t *BwdFilterAlg, int *InCh, int *IR, int *SR)
{
	//----------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This function allocates and initializes the tensor, filter, convolution descriptors, and convolution algorithms of the cudnn.lib library
	required to setup and initilize the convolutional layers.
	*/

	/****Argument List****/
	/*

	Desc_X:-        tensor descriptor for the convolutional layer input.
	Desc_Y:-        tensor descriptor for the convolutional layer output.
	Desc_W:-        filter descriptor for the convolutional layer weights.
	Conv_Desc:-     convolution descriptor for the convolutional layer.
	Desc_Xs:-       tensor descriptor for the convolutional layer input of a residual connection.
	Desc_Ys:-       tensor descriptor for the convolutional layer output of a residual connection.
	Desc_Ws:-       filter descriptor for the convolutional layer weights of a residual connection.
	Conv_s_Desc:-   convolution descriptor for the convolutional layer of a residual connection.
	FwdAlg:-        decides the convolution algorithm used in the forward pass.
	BwdDataAlg:-    decides the convolution algorithm used to backpropagate the error signal through the convolutional layer.
	BwdFilterAlg:-  decides the convolution algorithm used to update the convolutional layer parameters.
	InCh:-          number the input channels to the convolutional layer
	IR:-            input channel size.

	*/
	//----------------------------------------------------------------------------------------------------------------------------------------------

	cudnnStatus_t dnnStatus;

	for (int i = 0; i < CL; i++)
	{
		dnnStatus = cudnnCreateTensorDescriptor(Desc_X + i);
		dnnStatus = cudnnCreateTensorDescriptor(Desc_Y + i);
		dnnStatus = cudnnCreateFilterDescriptor(Desc_W + i);
		dnnStatus = cudnnCreateConvolutionDescriptor(Conv_Desc + i);

		dnnStatus = cudnnSetTensor4dDescriptor(Desc_X[i], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BatchSize, InCh[i], IR[i], IR[i]);
		dnnStatus = cudnnSetFilter4dDescriptor(Desc_W[i], CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, OutCh[i], InCh[i], Kr[i], Kr[i]);

		dnnStatus = cudnnSetConvolution2dDescriptor(Conv_Desc[i], PAD[i], PAD[i], STRIDE[i], STRIDE[i], 1, 1, CUDNN_CROSS_CORRELATION);

		int batch, ch, h, w;
		dnnStatus = cudnnGetConvolution2dForwardOutputDim(Conv_Desc[i], Desc_X[i], Desc_W[i], &batch, &ch, &h, &w);

		dnnStatus = cudnnSetTensor4dDescriptor(Desc_Y[i], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, ch, h, w);

	}

	//--------------------------------------------------------------------------

	dnnStatus = cudnnCreateConvolutionDescriptor(Conv_s_Desc);
	dnnStatus = cudnnSetConvolution2dDescriptor(*Conv_s_Desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION);

	for (int i = 0; i < CL - JMP; i += JMP)
	{
		if (OutCh[i] != OutCh[i + JMP])
		{
			dnnStatus = cudnnCreateTensorDescriptor(Desc_Xs + i);
			dnnStatus = cudnnCreateTensorDescriptor(Desc_Ys + i);
			dnnStatus = cudnnCreateFilterDescriptor(Desc_Ws + i);

			dnnStatus = cudnnSetTensor4dDescriptor(Desc_Xs[i], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BatchSize, OutCh[i], SR[i], SR[i]);
			dnnStatus = cudnnSetFilter4dDescriptor(Desc_Ws[i], CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, OutCh[i + JMP], OutCh[i], 1, 1);

			int batch, ch, h, w;
			dnnStatus = cudnnGetConvolution2dForwardOutputDim(*Conv_s_Desc, Desc_Xs[i], Desc_Ws[i], &batch, &ch, &h, &w);
			dnnStatus = cudnnSetTensor4dDescriptor(Desc_Ys[i], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, ch, h, w);
		}
	}

	//--------------------------------------------------------------------------

	for (int i = 0; i < CL; i++)
	{
		if (STRIDE[i] == 1 && Kr[i] == 3)
		{
			FwdAlg[i] = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
			BwdFilterAlg[i] = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED;
			BwdDataAlg[i] = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD;
		}
		else
		{
			FwdAlg[i] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
			BwdFilterAlg[i] = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
			BwdDataAlg[i] = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
		}
	}
}

//============================================================================================================================================================

void AllocateParamGPUMemory(float ***P_W, float ***P_V, float ***P_DW, float ***P_Ws, float ***P_Vs, float ***P_DWs, float ***P_X, float ***P_Y, float ***P_Param, float ***P_DParam, float ***P_ParamV, float ***P_SMU, float ***P_Derv, float **P_WF, float **P_VF, float **P_DWF, float **P_YF, float **P_Yv, int ***P_Indx, bool ***P_F, int *WSize, int *WsSize, int *Xr, int *Xc, int *Yr, int *Yc, int *SR)
{
	//----------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This function allocates the GPU memory required to hold the network parameters including the current derivatives and the running averages
	of the squared derivatives.
	*/

	/****Argument List****/
	/*

	P_W:-      pointer to a GPU buffer that will be allocated to hold the weights of all convolutional layers.
	P_V:-      pointer to a GPU buffer that will be allocated to hold the running average of the squared derivatives of the weights of all convolutional layers.
	P_DW:-     pointer to a GPU buffer that will be allocated to hold the current derivatives of the weights of all convolutional layers.
	P_Ws:-     pointer to a GPU buffer that will be allocated to hold the weights of all convolutional layers for residual connections.
	P_Vs:-     pointer to a GPU buffer that will be allocated to hold the running average of the squared derivatives of the weights of all convolutional layers for residual connections.
	P_DWs:-    pointer to a GPU buffer that will be allocated to hold the current derivatives of the weights of all convolutional layers for residual connections.
	P_X:-      pointer to a GPU buffer that will be allocated to hold the input channels of all convolutional layers.
	P_Y:-      pointer to a GPU buffer that will be allocated to hold the output channels of all convolutional layers.
	P_Param:-  pointer to a GPU buffer that will be allocated to hold the BN trainable parameters.
	P_ParamV:- pointer to a GPU buffer that will be allocated to hold the running average of the squared derivatives of the BN trainable parameters.
	P_DParam:- pointer to a GPU buffer that will be allocated to hold the current derivatives of the BN trainable parameters.
	P_SMU:-    pointer to a GPU buffer that will be allocated to hold the current means and variances of all BN stages.
	P_Derv:-   pointer to a GPU buffer that will be allocated to hold an intermediate values used to backpropagate the error signal through BN stages.
	P_WF:-     pointer to a GPU buffer that will be allocated to hold the weights of the FC output layer.
	P_VF:-     pointer to a GPU buffer that will be allocated to hold the running average of the squared derivatives of the weights of the FC output layer.
	P_DWF:-    pointer to a GPU buffer that will be allocated to hold the current derivatives of the weights of the FC output layer.
	P_YF:-     pointer to a GPU buffer that will be allocated to hold the outputs of the output layer.
	P_F:-      pointer to a GPU buffer that will be allocated to hold the signs of the combined convolutional layer/residual connection output.
	P_Indx:-   pointer to a GPU buffer that will be allocated to hold the indices of the maximum values of the maxpooling stages.

	*/
	//----------------------------------------------------------------------------------------------------------------------------------------------

	cudaError_t cudaStatus;

	float **d_W, **d_V, **d_DW, **d_X, **d_Y, **d_Param, **d_DParam, **d_ParamV, **d_SMU, **d_Derv, *d_WF, *d_VF, *d_DWF, *d_YF, *d_Yv;
	
	d_W = new float*[CL];
	d_V = new float*[CL];
	d_DW = new float*[CL];
	d_X = new float*[CL];
	d_Y = new float*[CL];
	d_Param = new float*[CL];
	d_ParamV = new float*[CL];
	d_DParam = new float*[CL];
	d_SMU = new float*[CL];
	d_Derv = new float*[CL];

	for (int i = 0; i < CL; i++)
	{
		cudaStatus = cudaMalloc(d_W + i, sizeof(float)*WSize[i]);
		cudaStatus = cudaMalloc(d_V + i, sizeof(float)*WSize[i]);
		cudaStatus = cudaMalloc(d_DW + i, sizeof(float)*WSize[i]);
		cudaStatus = cudaMalloc(d_X + i, sizeof(float)*Xr[i] * Xc[i]);
		cudaStatus = cudaMalloc(d_Y + i, sizeof(float)*Yr[i] * Yc[i]);
		cudaStatus = cudaMalloc(d_Param + i, sizeof(float) * 2 * OutCh[i]);
		cudaStatus = cudaMalloc(d_ParamV + i, sizeof(float) * 2 * OutCh[i]);
		cudaStatus = cudaMalloc(d_DParam + i, sizeof(float) * 2 * OutCh[i]);
		cudaStatus = cudaMalloc(d_SMU + i, sizeof(float) * 2 * OutCh[i] * BatchSize);
		cudaStatus = cudaMalloc(d_Derv + i, sizeof(float) * 2 * OutCh[i]);
		cudaStatus = cudaMemset(d_V[i], 0, sizeof(float)*WSize[i]);
		cudaStatus = cudaMemset(d_ParamV[i], 0, sizeof(float) * 2 * OutCh[i]);
	}

	cudaStatus = cudaMalloc(&d_YF, sizeof(float)*Out1*BatchSize);
	cudaStatus = cudaMalloc(&d_Yv, sizeof(float)*Out1*BatchSize);
	cudaStatus = cudaMalloc(&d_WF, sizeof(float)*Out1*(In1 + 1));
	cudaStatus = cudaMalloc(&d_VF, sizeof(float)*Out1*(In1 + 1));
	cudaStatus = cudaMalloc(&d_DWF, sizeof(float)*Out1*(In1 + 1));
	cudaStatus = cudaMemset(d_VF, 0, sizeof(float) * Out1*(In1 + 1));

	*P_W = d_W; *P_V = d_V; *P_DW = d_DW; *P_X = d_X; *P_Y = d_Y; *P_Param = d_Param; *P_DParam = d_DParam; *P_ParamV = d_ParamV; *P_SMU = d_SMU; *P_Derv = d_Derv; *P_WF = d_WF; *P_VF = d_VF; *P_DWF = d_DWF; *P_YF = d_YF; *P_Yv = d_Yv;

	//--------------------------------------------------------------------------

	float **d_Ws, **d_Vs, **d_DWs;

	d_Ws = new float*[CL];
	d_Vs = new float*[CL];
	d_DWs = new float*[CL];

	for (int i = 0; i < CL - JMP; i += JMP)
	{
		if (OutCh[i] != OutCh[i + JMP])
		{
			WsSize[i] = OutCh[i] * OutCh[i + JMP];

			cudaStatus = cudaMalloc(d_Ws + i, sizeof(float)*WsSize[i]);
			cudaStatus = cudaMalloc(d_Vs + i, sizeof(float)*WsSize[i]);
			cudaStatus = cudaMalloc(d_DWs + i, sizeof(float)*WsSize[i]);
			cudaStatus = cudaMemset(d_Vs[i], 0, sizeof(float)*WsSize[i]);
		}
	}

	*P_Ws = d_Ws; *P_Vs = d_Vs; *P_DWs = d_DWs;

	//--------------------------------------------------------------------------

	int **d_Indx; bool **d_F;

	d_Indx = new int*[CL];

	for (int i = 0; i < CL; i++)
	{
		if (PoolType[i] == 1)
		{
			cudaStatus = cudaMalloc(d_Indx + i, sizeof(int)*SR[i] * SR[i] * BatchSize * OutCh[i]);
		}
	}

	d_F = new bool*[CL];
	for (int i = JMP; i < CL; i += JMP)
	{
		cudaStatus = cudaMalloc(d_F + i, sizeof(bool)*Yr[i] * Yc[i]);
	}

	*P_Indx = d_Indx; *P_F = d_F;

}

//============================================================================================================================================================

void AllocateGPUMemory_ExtraOutputLayer(float **P_WF2, float **P_VF2, float **P_DWF2, float **P_YF2, float **P_XC, float **P_mse2, float **P_count2, int *Xr, int *Xc)
{
	//----------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This function allocates the GPU memory required for the extra output layer.
	*/

	/****Argument List****/
	/*
	
	P_WF2 :-   pointer to a GPU buffer that will be allocated to hold the weights of the extra FC output layer.
    P_VF2 : -  pointer to a GPU buffer that will be allocated to hold the running average of the squared derivatives of the weights of the extra FC output layer.
    P_DWF2 : - pointer to a GPU buffer that will be allocated to hold the current derivatives of the weights of the extra FC output layer.
	P_YF2 : -  pointer to a GPU buffer that will be allocated to hold the outputs of the extra output layer.
	P_Xc : -   Pointer to a GPU buffer that will be allocated to hold back-propagated error from the extra output layer before combining it with back-propagated error signal from the main output layer.
	Xr[K_Cat + 1] * Xc[K_Cat + 1]:- the size of the *P_Xc buffer.

	*/
	//----------------------------------------------------------------------------------------------------------------------------------------------

	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc(P_YF2, sizeof(float)*Out2*BatchSize);
	cudaStatus = cudaMalloc(P_XC, sizeof(float)*Xr[K_Cat + 1] * Xc[K_Cat + 1]);

	cudaStatus = cudaMalloc(P_WF2, sizeof(float)*Out2*(In2 + 1));
	cudaStatus = cudaMalloc(P_VF2, sizeof(float)*Out2*(In2 + 1));
	cudaStatus = cudaMalloc(P_DWF2, sizeof(float)*Out2*(In2 + 1));
	cudaStatus = cudaMemset(*P_VF2, 0, sizeof(float) * Out2*(In2 + 1));

	float *d_mse2, *d_count2;

	cudaStatus = cudaMalloc(&d_mse2, sizeof(float)*BatchSize);
	cudaStatus = cudaMalloc(&d_count2, sizeof(float));

	*P_mse2 = d_mse2; *P_count2 = d_count2;
}

//============================================================================================================================================================

void AllocateAuxiliaryGPUMemory(float ***P_SMUs, float **P_YY, float **P_Y0, float **P_ws, float **P_rand1, float **P_randRGB, float **P_Cropf, unsigned int **P_Crop, float **P_mse, float **P_count, int **P_Indx1, int ***PIndx, int *PStart, int *Yr, int *Yc, int NParts)
{
	//----------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This function allocates auxiliary GPU memory buffers required to hold intermidiate results and support few cuda kernels.
	*/

	/****Argument List****/
	/*

	P_YY:-      pointer ot a GPU buffer that will be allocated to hold intermidiate output channels between BN and maxpooling.
	P_Y0:-      pointer ot a GPU buffer that will be allocated to hold the the inputs to output FC layer.
	P_ws:-      pointer ot a GPU buffer that will be allocated to hold temp GPU memory that will be used by the cudnn convolution algorithms.
	P_rand1:-   pointer ot a GPU buffer that will be allocated to hold random normal values that will be used by the RGBRandPCA cuda kernel for colour augmentation.
	P_randRGB:- pointer ot a GPU buffer that will be allocated to hold stochastic values generated by the RGBRandPCA cuda kernel for colour augmentation.
	P_Crop:-    pointer ot a GPU buffer that will be allocated to hold random integer values used to decide croping positions and probability of horizontal flipping for data augmentation.
	P_Cropf:-   pointer ot a GPU buffer that will be allocated to hold the input channels of all convolutional layers.
	P_mse:-     pointer ot a GPU buffer that will be allocated to accumulate the mean square error for the training/validation sets.
	P_count:-   pointer ot a GPU scalar to compute the number of correctly classified training/validation images.
	P_Indx1:-   pointer ot a GPU buffer that will be allocated to hold image indices for a data segement generated by the resuffel algorithm.
	PIndx:-     pointer ot a buffer that will be allocated to be used by the resuffel function.
	P_SMUs:-    pointer ot a GPU buffer that will be allocated to hold the fixed means and variances of all BN stages that will be used in the inference stage.

	*/
	//----------------------------------------------------------------------------------------------------------------------------------------------

	cudaError_t cudaStatus;

	float *d_YY, *d_Y0, *d_ws;

	int size = 0;
	for (int i = 0; i < CL; i++)
	{
		if (Yr[i] * Yc[i] > size)size = Yr[i] * Yc[i];
	}

	cudaStatus = cudaMalloc(&d_YY, sizeof(float)*(size + BatchSize));
	cudaStatus = cudaMalloc(&d_Y0, sizeof(float)*(In1 + 1)*BatchSize);

	float *TempBuffer = new float[(In1 + 1)*BatchSize];
	for (int i = 0; i < (In1 + 1)*BatchSize; i++)TempBuffer[i] = 1.0f;
	cudaStatus = cudaMemcpy(d_Y0, TempBuffer, sizeof(float)*(In1 + 1)*BatchSize, cudaMemcpyDefault);

	cudaStatus = cudaMalloc(&d_ws, sizeof(float)*ISize1 *BatchSize * OutCh[0]);

	*P_YY = d_YY; *P_Y0 = d_Y0; *P_ws = d_ws;

	//--------------------------------------------------------------------------

	unsigned int *d_Crop;
	float *d_rand1, *d_randRGB, *d_Cropf;

	cudaStatus = cudaMalloc(&d_Crop, sizeof(int) * 4 * TrainSizeM);
	cudaStatus = cudaMalloc(&d_rand1, sizeof(float) * 3 * TrainSizeM);
	cudaStatus = cudaMalloc(&d_randRGB, sizeof(float) * 3 * TrainSizeM);
	cudaStatus = cudaMalloc(&d_Cropf, sizeof(float) * 2 * TrainSizeM);

	*P_rand1 = d_rand1; *P_randRGB = d_randRGB; *P_Cropf = d_Cropf; *P_Crop = d_Crop;

	float *d_mse, *d_count;

	cudaStatus = cudaMalloc(&d_mse, sizeof(float)*BatchSize);
	cudaStatus = cudaMalloc(&d_count, sizeof(float));

	*P_mse = d_mse; *P_count = d_count;

	//--------------------------------------------------------------------------

	int **Indx;
	int *d_Indx1;

	int MaxPart = 0;
	for (int i = 0; i < NParts; i++)
		if ((PStart[i + 1] - PStart[i]) > MaxPart) MaxPart = (PStart[i + 1] - PStart[i]);

	Indx = new int*[NParts];
	for (int i = 0; i < NParts; i++)
	{
		Indx[i] = new int[PStart[i + 1] - PStart[i]];
		int *ptemp = Indx[i];
		for (int j = 0; j < (PStart[i + 1] - PStart[i]); j++) ptemp[j] = j;
	}

	cudaStatus = cudaMalloc(&d_Indx1, sizeof(int)*MaxPart);

	*P_Indx1 = d_Indx1; *PIndx = Indx;

	float **d_SMUs = new float*[CL];
	for (int i = 0; i < CL; i++)
	{
		cudaStatus = cudaMalloc(d_SMUs + i, sizeof(float) * 2 * OutCh[i]);
		cudaStatus = cudaMemset(d_SMUs[i], 0, sizeof(float) * 2 * OutCh[i]);
	}

	*P_SMUs = d_SMUs;

}

//============================================================================================================================================================

void InitializeConvLayerParam(int *IR, int *CR, int *SR, int *InCh, int *Xr, int *Xc, int *Yr, int *Yc, int *WSize, float **Buffer)
{
	//----------------------------------------------------------------------------------------------------------------------------------------------
	/*
	The InitializeConvLayerParam() initializes the following variables which define the structure of the convolutional layers.
	*/

	/**** Argument List****/
	/*

	CL :-    The number of convolutional layers.
	Xr[i], Xc[i] :- define the total size of the input channels for layer i, total input channel size of all images in the batch is Xr[i]×Xc[i].
	Yr[i], Yc[i] :- define the total size of the output channels for layer i, total output channel size of all images in the batch is Yr[i]×Yc[i].
	WSize[i] :-     Number of weights in convolutional layer i.
	IR[i] :-        the height/width of a single square input channel for convolutional layer i.
	CR[i] :-        the height/width of a single square output channel before maxpooling for convolutional layer i.
	SR[i] :-        the height/width of a single square output channel after maxpooling for convolutional layer i.
	InCh[i] :-      number of input channels for convolutional layer i.
	OutCh[i] :-     number of output channels for convolutional layer i.
	In1:-           number on inputs for the FC output layer.
	TEMP:-          temporary main memory buffer used to swap data between main memory and GPU memory.

	*/
	//----------------------------------------------------------------------------------------------------------------------------------------------
	
	InCh[0] = InCh1;
	IR[0] = IR1;
	for (int i = 0; i < CL; i++)
	{
		if (i == 0) CR[i] = (IR[i] + 2 * PAD[i] - Kr[i] + 2) / 2;
		else        CR[i] = IR[i] + 2 * PAD[i] - Kr[i] + 1;

		SR[i] = (PoolType[i] == 0) ? CR[i] : (CR[i] + Sr1[i] - 1) / Sr1[i];
		if (i < CL - 1)
		{
			InCh[i + 1] = OutCh[i];
			IR[i + 1] = SR[i];
		}

		WSize[i] = OutCh[i] * InCh[i] * Kr[i] * Kr[i];

		Xr[i] = IR[i] * IR[i];
		Xc[i] = InCh[i] * BatchSize;

		Yr[i] = CR[i] * CR[i];
		Yc[i] = OutCh[i] * BatchSize;
	}

	int Buffsize = 0;
	for (int i = 0; i < CL; i++)
	{		
		if (Buffsize < WSize[i])Buffsize = WSize[i];
	}

	if (Buffsize < Out1*(In1 + 1)) Buffsize = Out1*(In1 + 1);

	*Buffer = new float[Buffsize];
}

//============================================================================================================================================================

void ParameterInitialization(float **d_W, float **d_Ws, float *d_WF, float **d_Param, float *Buffer, int *WSize, int *WsSize)
{
	//----------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This function inializes the trainable parameters of the network. All weight are initialized using Kaiming initialization. The BN trainable
	parameters are initialized to 0 and 1.
	*/

	/**** Argument List ****/
	/*

	d_W :-     the weights of all conv layers.
	d_Ws :-    the weights of the conv layers of the residual connections.
	d_WF :-    the weights of the FC output layer.
	d_Param :- the trainable parameters of the batch normalization layers.

	*/
	//----------------------------------------------------------------------------------------------------------------------------------------------

	cudaError_t cudaStatus;
	
	extern mt19937 gen;
	normal_distribution<float>distr(0.0f, 1.0f);

	for (int i = 0; i < CL; i++)
	{
		float sgma = (i == 0) ? sqrtf(1.0f / (InCh1 * Kr[i] * Kr[i])) : sqrtf(2.0f / (OutCh[i - 1] * Kr[i] * Kr[i]));
		for (int j = 0; j < WSize[i]; j++) Buffer[j] = sgma * distr(gen);
		cudaStatus = cudaMemcpy(d_W[i], Buffer, sizeof(float)*WSize[i], cudaMemcpyHostToDevice);
	}

	for (int i = 0; i < CL - JMP; i += JMP)
	{
		if (OutCh[i] != OutCh[i + JMP])
		{
			float sgma = sqrtf(2.0f / (OutCh[i]));
			for (int j = 0; j < WsSize[i]; j++)
			{
				Buffer[j] = sgma*distr(gen);
			}
			cudaStatus = cudaMemcpy(d_Ws[i], Buffer, sizeof(float)*WsSize[i], cudaMemcpyHostToDevice);		
		}
	}

	float sgma = sqrtf(1.0f / In1);

	for (int i = 0; i < (In1 + 1); i++)
		for (int j = 0; j < Out1; j++)
		{
			if (i == In1) Buffer[Out1*i + j] = 0.0f;
			else               Buffer[j + Out1*i] = sgma*distr(gen);
		}
	cudaStatus = cudaMemcpy(d_WF, Buffer, sizeof(float)*Out1*(In1 + 1), cudaMemcpyHostToDevice);


	float a1 = 1.0f, a2 = 0.0f;
	for (int i = 0; i < 2 * OutCh[CL - 1]; i++)
	{
		if (i % 2 == 0)  Buffer[i] = a1;
		else             Buffer[i] = a2;
	}

	for (int i = 0; i < CL; i++)
		cudaStatus = cudaMemcpy(d_Param[i], Buffer, sizeof(float) * 2 * OutCh[i], cudaMemcpyHostToDevice);
	
}

//============================================================================================================================================================

void ParameterInitialization_ExtraOutputLayer(float *d_WF2, float *Buffer)
{
	//----------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This function initializes the weights of the extra output layer.
	*/

	/**** Argument List ****/
	/*

	d_WF2 :-   the weights  of the extra FC output layer.
	Buffer :-  temporary main memory buffer. 

    */
	//----------------------------------------------------------------------------------------------------------------------------------------------

	cudaError_t cudaStatus;

	extern mt19937 gen;
	normal_distribution<float>distr(0.0f, 1.0f);

	float sgma = sqrtf(1.0f / In2);

	for (int i = 0; i < (In2 + 1); i++)
		for (int j = 0; j < Out2; j++)
		{
			if (i == In2) Buffer[j + Out2*i] = 0.0f;
			else               Buffer[j + Out2*i] = sgma*distr(gen);
		}
	cudaStatus = cudaMemcpy(d_WF2, Buffer, sizeof(float)*Out2*(In2 + 1), cudaMemcpyHostToDevice);
}

//============================================================================================================================================================

void InitializeCudaKernels(dim3 *gridSizeP, dim3 *gridSizeBN1, dim3 *gridSizeBN11, dim3 *gridSizeBN2, dim3 *gridSizeAddA, dim3 *P_gridSizeRGB, dim3 *P_gridSize_Crop, dim3 *P_gridSizePA, int *gridSizeAddYB, int *gridSizeAddWs, int *gridSizeAddW, int *P_gridSizeAddWF, int *SR, int *Yr, int *Yc, int *WSize, int *WsSize)
{
	//----------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This function intializes the following grid sizes of the main cuda kernels used to implement a deep residual CNN.
	*/

	/**** Argument List ****/
	/*

	gridSizeP[i] :-      the grid size for the maxpooling kernel of convolutional layer i.
	gridSizeBN1[i] :-    the grid size for the fisrt stage of the forward pass and the second stage of the backward pass kernel for BN of convolutional layer i.
	gridSizeBN11[i] :-   the grid size for the fisrt stage of the forward pass kernel for BN of convolutional layer i.
	gridSizeBN2[i] :-    the grid size for the second stage of the forward pass and the first stage of the backward pass kernel for BN of convolutional layer i.
	gridSizeAddA[i] :-   the grid size for the cuda kernel used to update the BN trainable parameters at layer i.
	P_gridSizeRGB :-     the grid size for the cuda kernel used to generate stochastic values used for colour augmentation.
	P_gridSize_Crop :-   the grid size for the data augmentation kernel.
	P_gridSizePA :-      the grid size for the global average pooling used after the last convolutional layer.
	gridSizeAddYB[i] :-  the grid size used by the AddMatrix kernel which adds two matrecies with equal sizes.
	gridSizeAddWs[i] :-  the grid size for the cuda kernel used to update the weights of the residual connection at layer i.
	gridSizeAddW[i] :-   the grid size for the cuda kernel used to update the weights of convolutional layer i.
	gridSizeAddWF :-     the grid size for the cuda kernel used to update the weights of FC output layer.

	*/
	//----------------------------------------------------------------------------------------------------------------------------------------------

	for (int i = 0; i < CL; i++)
	{
		int sn = SR[i] * SR[i];

		dim3 gsizep((sn + BLOCKSIZE1 - 1) / BLOCKSIZE1, OutCh[i], BatchSize);
		gridSizeP[i] = gsizep;

		dim3 gsizebn1(OutCh[i], BatchSize, 1);
		gridSizeBN1[i] = gsizebn1;

		dim3 gsizebn11((OutCh[i] + BLOCKSIZE4 - 1) / BLOCKSIZE4, 1, 1);
		gridSizeBN11[i] = gsizebn11;

		dim3 gsizebn2((Yr[i] + BLOCKSIZE4 - 1) / BLOCKSIZE4, OutCh[i], BatchSize);
		gridSizeBN2[i] = gsizebn2;
	}

	//--------------------------------------------------------------------------

	for (int i = 0; i < CL - JMP; i += JMP)
	{
		gridSizeAddYB[i] = GridSizeAC(BLOCKSIZE1, Yr[i + JMP] * Yc[i]);
		if (OutCh[i] != OutCh[i + JMP])
			gridSizeAddWs[i] = GridSizeAC(BLOCKSIZE1, WsSize[i]);
	}

	for (int i = 0; i < CL; i++)
		gridSizeAddW[i] = GridSizeAC(BLOCKSIZE1, WSize[i]);

	for (int i = 0; i < CL; i++)
	{
		dim3 gsizeaddA((2 * OutCh[i] + BLOCKSIZE1 - 1) / BLOCKSIZE1, 1, 1);
		gridSizeAddA[i] = gsizeaddA;
	}

	//--------------------------------------------------------------------------

	dim3 gridSizePA(OutCh[CL - 1], BatchSize, 1);
	*P_gridSizePA = gridSizePA;

	dim3 gridSizeRGB((TrainSizeM + BLOCKSIZE1 - 1) / BLOCKSIZE1, 1, 1);
	*P_gridSizeRGB = gridSizeRGB;

	dim3 gridSize_Crop((ISize1 + BLOCKSIZE1 - 1) / BLOCKSIZE1, BatchSize, 1);
	*P_gridSize_Crop = gridSize_Crop;

	int gridSizeAddWF = GridSizeAC(BLOCKSIZE1, Out1*(In1 + 1));
	*P_gridSizeAddWF = gridSizeAddWF;

}

//============================================================================================================================================================

void InitializeCudaKernels_ExtraOutputLayer(dim3 *P_gridSizePA_Cat, int *P_gridSizeAdd_Cat, int *P_gridSizeAddWF2, int *Xr, int *Xc)
{
	//----------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This function initializes the following grid sizes of the main cuda kernels used to implement a deep residual CNN.
	*/

	/**** Argument List ****/
	/*
	
	P_gridSizePA_Cat :-  the grid size for the global average pooling used after convolutional layer 27 (K_Cat = 26) and before the extra output layer.
	P_gridSizeAdd_Cat :- the grid size for the Add_Mtx cuda kernel used to combine the back-propagated error signals from the main and the extra output layer. 
	gridSizeAddWF2 :-    the grid size for the cuda kernel used to update the weights of the extra FC output layer.
	Xr[K_Cat + 1] * Xc[K_Cat + 1] :- is the size of the combined back-propagated error matrix.

	*/
	//----------------------------------------------------------------------------------------------------------------------------------------------

	dim3 gridSizePA_Cat(OutCh[K_Cat], BatchSize, 1);
	int gridSizeAdd_Cat = GridSizeAC(BLOCKSIZE1, Xr[K_Cat + 1] * Xc[K_Cat + 1]);
	int gridSizeAddWF2 = GridSizeAC(BLOCKSIZE1, Out2*(In2 + 1));

	*P_gridSizePA_Cat = gridSizePA_Cat;
	*P_gridSizeAdd_Cat = gridSizeAdd_Cat;
	*P_gridSizeAddWF2 = gridSizeAddWF2;
}

//============================================================================================================================================================

void PrintIterResults(FILE *out, float *d_mse, float *d_count, int *PStart, int NParts, int VParts, int iter)
{
	cudaError_t cudaStatus;
	
	extern float ErrorT, ErrorV, MSE;
	extern int CountT, CountV;

	float *mse = new float[BatchSize];
	float Error, Countp;
	
	cudaStatus = cudaMemcpyAsync(mse, d_mse, sizeof(float)*BatchSize, cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpyAsync(&Countp, d_count, sizeof(float), cudaMemcpyDeviceToHost);

	int p = iter%NParts;
	if (p == 0)
	{
		ErrorT = 0.0f; CountT = 0;
		ErrorV = 0.0f; CountV = 0;
	}

	int psize = PStart[p + 1] - PStart[p];
	
	Error = 0.0f;
	for (int ii = 0; ii < BatchSize; ii++)
		Error += mse[ii];
	MSE = Error / float(psize);
	
	cout << "mse= " << Error / float(psize) << "     r = " << Countp / float(psize) << "      c = " << Countp <<"       " << iter << endl;

	if (p < NParts - VParts) { CountT += int(Countp); ErrorT += Error; }
	else                { CountV += int(Countp); ErrorV += Error; }
	if (p == NParts - 1)
	{
		if (VParts == 0)
		{
			cout << endl << "MSE = " << float(ErrorT) / float(TrainSizeM) << "   RATE = " << float(CountT) / float(TrainSizeM) << endl << endl;
			fprintf(out, "%f\t%f\n", float(ErrorT) / float(TrainSizeM), float(CountT) / float(TrainSizeM));
		}
		else
		{
			cout << endl << "MSE = " << float(ErrorT) / float(TrainSizeM) << "   RATE = " << float(CountT) / float(TrainSizeM) << "   MSE = " << float(ErrorV) / float(ValSize) << "   Val = " << float(CountV) / float(ValSize) << "   " << (iter / NParts) + 1 << endl << endl;
			fprintf(out, "%f\t%f\t%f\t%f\n", float(ErrorT) / float(TrainSizeM), float(ErrorV) / float(ValSize), float(CountT) / float(TrainSizeM), float(CountV) / float(ValSize)); 
			fflush(out);
		}
	}

}

//============================================================================================================================================================

void InitializeMultiCropInference(int **P_MTX)
{
	//----------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This function calculates the cropping positions and scales for all test images in the testset. Each test images is scaled to a number of scales
	equal to ScaleSteps, and these scales are proportional to the shorter side of the image. At each scale the number of cropes is equal to
	EpochTs/ScaleSteps where EpochTs is equal to the total number of crops per image. Those cropping positions will be calcuated in a way that is 
	equally spaced thoughout each image scale. For each croping position there are 3 integer values, the (x,y) cropping coordinate and the scale at
	which that crop was taken. Therefore for a single test image there is a total of 3*EpochTs integer values that will stored in the P_MTX buffer. 
	*/


	//----------------------------------------------------------------------------------------------------------------------------------------------

	cudaError_t cudaStatus;

	extern mt19937 gen;
	uniform_int_distribution<int> distr_int(0, 1000000000);
	uniform_real_distribution<float> distr_uniform(0.0f, 1.0f);

	extern char DataFloder[];
	char FileName[128];

	strcpy_s(FileName, 128, DataFloder); strcat_s(FileName, 128, "cordinatesVal.txt");
	FILE *in1; fopen_s(&in1, FileName, "rb");
	unsigned int *Height = new unsigned int[TestSizeM];
	unsigned int *Width = new unsigned int[TestSizeM];
	size_t numread1 = fread(Height, sizeof(unsigned int), TestSizeM, in1);
	size_t numread2 = fread(Width, sizeof(unsigned int), TestSizeM, in1);

	int *MTX = new int[TestSize*EpochTs * 3];
	int *MTX2 = new int[EpochTs * 3];

	cudaStatus = cudaMalloc(P_MTX, sizeof(int)*TestSize*EpochTs * 3);
	int *d_MTX = *P_MTX;

	for (int i = 0; i < TestSize; i++)
	{
		int steps = EpochTs / ScaleSteps;
		int k = 0;
		float H = float(Height[i]);
		float W = float(Width[i]);
		int *SMTX;
		SMTX = MTX2;

		for (int scale = Scale1; scale <= Scale2; scale += ScaleInc)
		{
			float c, x, y, x1, y1, f, extra, distx, disty;
			int Hs, Ws;

			if (H > W) { Ws = scale; Hs = scale*int(H / W); }
			else { Hs = scale; Ws = scale*int(W / H); }

			distx = float(Hs - IR1), disty = float(Ws - IR1);

			c = distx / disty;
			y = sqrtf(steps / c);
			x = c*y;
			x1 = floorf(x);
			y1 = floorf(y);
			extra = steps - x1*y1;

			bool cond = (x / x1) >= (y / y1);
			if (cond && extra > y1) { x1++; extra -= y1; }
			if (!cond && extra > x1) { y1++; extra -= x1; }

			float xs1, xs2, ys1, ys2;
			if (x1*y1 < steps)
			{
				if (x / x1 > y / y1) { f = 1.0f; }
				else { f = 0.0f; }

				if (f == 1)
				{
					xs1 = (y1*distx) / (x1*y1 + extra);
					xs2 = distx - x1*xs1;
					ys1 = disty / y1;
					ys2 = disty / extra;
					for (int c = 0; c < y1; c++)
						for (int r = 0; r < x1; r++)
						{
							SMTX[k] = scale;
							SMTX[k + 1] = int(0.5f*xs1 + r*xs1);
							SMTX[k + 2] = int(0.5f*ys1 + c*ys1);
							k = k + 3;
						}

					for (int c = 0; c < extra; c++)
					{
						SMTX[k] = scale;
						SMTX[k + 1] = int(xs1*x1 + 0.5f*xs2);
						SMTX[k + 2] = int(0.5f*ys2 + c*ys2);
						k = k + 3;
					}
				}
				else
				{
					ys1 = (x1*disty) / (x1*y1 + extra);
					ys2 = disty - y1*ys1;
					xs1 = distx / x1;
					xs2 = distx / extra;
					for (int c = 0; c < y1; c++)
						for (int r = 0; r < x1; r++)
						{
							SMTX[k] = scale;
							SMTX[k + 1] = int(0.5f*xs1 + r*xs1);
							SMTX[k + 2] = int(0.5f*ys1 + c*ys1);
							k = k + 3;
						}

					for (int r = 0; r < extra; r++)
					{
						SMTX[k] = scale;
						SMTX[k + 1] = int(0.5f*xs2 + r*xs2);
						SMTX[k + 2] = int(ys1*y1 + 0.5f*ys2);
						k = k + 3;
					}
				}
			}
			else
			{
				xs1 = distx / x1;
				ys1 = disty / y1;

				for (int c = 0; c < y1; c++)
					for (int r = 0; r < x1; r++)
					{
						SMTX[k] = scale;
						SMTX[k + 1] = int(0.5f*xs1 + r*xs1);
						SMTX[k + 2] = int(0.5f*ys1 + c*ys1);
						k = k + 3;
					}
			}

		}
		int *ptemp = MTX + i*EpochTs * 3;
		int *SSteps, *TCount, *Count, *randk;
		SSteps = new int[ScaleSteps];
		TCount = new int[ScaleSteps];
		Count = new int[ScaleSteps];
		randk = new int[ScaleSteps];
		
		for (int j = 0; j < ScaleSteps; j++)
		{
			randk[j] = j;
			TCount[j] = 0;
			Count[j] = steps;
			if (j == 0) SSteps[j] = 0; else SSteps[j] = SSteps[j - 1] + steps;
		}

		float complete = EpochTs / ScaleSteps;
		int ix = 0; k = 0;
		while (ix < EpochTs)
		{
			int x;
			if (k == 0)
			{
				for (int j = ScaleSteps - 1; j >= 1; j--)
				{
					x = distr_int(gen) % j;
					int temp = randk[x]; randk[x] = randk[j]; randk[j] = temp;
				}
			}
			int k1 = randk[k];
			float pro = (Count[k1] - TCount[k1]) / complete;
			float px = distr_uniform(gen);
			if (px < pro)
			{
				x = TCount[k1] + distr_int(gen) % (Count[k1] - TCount[k1]);
				ptemp[3 * ix] = SMTX[3 * SSteps[k1] + 3 * x];
				ptemp[3 * ix + 1] = SMTX[3 * SSteps[k1] + 3 * x + 1];
				ptemp[3 * ix + 2] = SMTX[3 * SSteps[k1] + 3 * x + 2];

				SMTX[3 * SSteps[k1] + 3 * x] = SMTX[3 * SSteps[k1] + 3 * TCount[k1]];
				SMTX[3 * SSteps[k1] + 3 * x + 1] = SMTX[3 * SSteps[k1] + 3 * TCount[k1] + 1];
				SMTX[3 * SSteps[k1] + 3 * x + 2] = SMTX[3 * SSteps[k1] + 3 * TCount[k1] + 2];
				TCount[k1]++;
				complete -= 1.0f / ScaleSteps;
				ix++;
			}
			k = (k + 1) % ScaleSteps;
		}

	}

	cudaStatus = cudaMemcpy(d_MTX, MTX, sizeof(int) * TestSize * EpochTs * 3, cudaMemcpyHostToDevice);

}

//============================================================================================================================================================

void PrintFinalResults(FILE *out1, float *d_Yss, float *d_mse, int *d_T)
{
	cudaError_t cudaStatus;

	float *mse = new float[BatchSize];
	float *Yss = new float[NumClasses*TestSizeM];
	int *T = new int[TestSizeM];

	cudaStatus = cudaMemcpy(mse, d_mse, sizeof(float)*BatchSize, cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(Yss, d_Yss, sizeof(float) * NumClasses*TestSizeM, cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(T, d_T, sizeof(int)*TestSizeM, cudaMemcpyDeviceToHost);

	float max1, max2;
	int indx1, indx2;
	int count[5] = {};

	int **COUNT;
	COUNT = new int*[NumClasses];
	for (int i = 0; i < NumClasses; i++)
	{
		COUNT[i] = new int[NumClasses];
		for (int j = 0; j < NumClasses; j++)COUNT[i][j] = 0;
	}

	count[0] = 0;
	for (int n = 0; n < 2; n++)
	{
		if (n >0)count[n] = count[n - 1];
		for (int i = 0; i < TestSizeM; i++)
		{
			max1 = -9.9e+30f;
			max2 = max1;
			for (int j = 0; j < NumClasses; j++)
			{
				if (Yss[i*NumClasses + j] > max2) { max2 = Yss[i*NumClasses + j]; indx2 = j; }
			}
			Yss[i*NumClasses + indx2] = -9.9e+30f;
			indx1 = T[i];
			if (indx1 == indx2) count[n]++;
			if (n == 0) COUNT[indx1][indx2]++;
		}

	}

	float Error = 0.0f;
	for (int i = 0; i < BatchSize; i++)
		Error += mse[i];

	float testsize = TestSizeM;

	cout << endl << "mse= " << Error / (EpochTs*testsize) << endl << endl;
	fprintf(out1, " \n mse = %f \n\n", Error / (EpochTs*testsize));
	for (int i = 0; i < 2; i++)
	{
		cout << endl << "conut = " << count[i] << "         ratio = " << count[i] / testsize << endl;
		fprintf(out1, " \n count = %d \t\t ratio = %f \n", count[i], count[i] / testsize);
	}

	fprintf(out1, "\n\n");
	cout << endl << endl;

	for (int i = 0; i < NumClasses; i++)
	{
		fprintf(out1, "\n");
		for (int j = 0; j < NumClasses; j++)
			fprintf(out1, "%d\t", COUNT[i][j]);
	}

	fprintf(out1, "\n\n");
	cout << endl << endl;

}

//============================================================================================================================================================

void ReshuffleImages(int *Indx1, int *PStart, int p)
{
	//----------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This function randomly reshuffles the images contained in a single data-segment that fits in half of the 3 RGB buffers.
	*/
	//----------------------------------------------------------------------------------------------------------------------------------------------

	extern mt19937 gen;
	uniform_int_distribution<int> distr_int(0, 1000000000);

	for (int j = (PStart[p + 1] - PStart[p] - 1); j >= 1; j--)
	{
		int r = distr_int(gen) % j;
		int temp = Indx1[j];
		Indx1[j] = Indx1[r];
		Indx1[r] = temp;
	}
}

//============================================================================================================================================================

void FreeTrainSpecificData(float **d_V, float **d_DW, float **d_Vs, float **d_DWs, float **d_DParam, float **d_ParamV, float **d_Derv, float *d_VF, float *d_DWF, float *d_count, float *d_rand1, float *d_randRGB, float *d_Cropf, int *d_Indx1, int **Indx, int *Indx1, unsigned int *d_Crop, int NParts)
{

	for (int i = 0; i < CL; i++)
	{
		float *temp = d_V[i]; cudaFree(temp);

		temp = d_DW[i];       cudaFree(temp);

		temp = d_Vs[i];       cudaFree(temp);

		temp = d_DWs[i];      cudaFree(temp);

		temp = d_DParam[i];   cudaFree(temp);

		temp = d_ParamV[i];   cudaFree(temp);

		temp = d_Derv[i];     cudaFree(temp);
	}

	cudaFree(d_Crop);     cudaFree(d_VF);     cudaFree(d_DWF);     cudaFree(d_Indx1);  
	cudaFree(d_count);  cudaFree(d_rand1);   cudaFree(d_randRGB); cudaFree(d_Cropf);
	
	//delete[] Indx1;
	
	/*for (int i = 0; i < NParts; i++)
	{
		delete[] Indx[i];
	}*/

}

//============================================================================================================================================================

void FreeRemainingMem(unsigned char *Red, unsigned char *Green, unsigned char *Blue, float **d_W, float **d_Ws, float **d_X, float **d_Y, float **d_Param, float **d_SMU, float **d_SMUs, int **d_Indx, bool **d_F, float *d_WF, float *d_YF, float *d_Yv, float *d_YY, float *d_Y0, float *d_ws, float *d_mse, unsigned int *d_HeightTr, unsigned int *d_WidthTr, size_t *d_StartTr, int *d_T, int *PStart, size_t *StartPart, size_t *PartSize, float *TEMP)
{
	
	cudaFreeHost(Red);
	cudaFreeHost(Green);
	cudaFreeHost(Blue);

	for (int i = 0; i < CL; i++)
	{
		float *temp = d_Y[i];   cudaFree(temp);
		temp = d_W[i];          cudaFree(temp);
		temp = d_Ws[i];         cudaFree(temp);
		temp = d_X[i];          cudaFree(temp); 
		temp = d_Y[i];          cudaFree(temp);
		temp = d_Param[i];      cudaFree(temp);
		temp = d_SMU[i];        cudaFree(temp);
		temp = d_SMUs[i];       cudaFree(temp);
		
		int *itemp = d_Indx[i];
		if (PoolType[i] == 1) cudaFree(itemp);

		bool *btemp = d_F[i];
		if (i >= JMP && i%JMP == 0) cudaFree(btemp); 

	}

	cudaFree(d_WF); cudaFree(d_YF); cudaFree(d_Yv); cudaFree(d_YY); cudaFree(d_Y0);  
	cudaFree(d_ws); cudaFree(d_mse);cudaFree(d_HeightTr); cudaFree(d_WidthTr); cudaFree(d_StartTr); cudaFree(d_T);

	delete[] PStart; delete[] StartPart; delete[] PartSize; delete[] TEMP;

}

//============================================================================================================================================================

static int GridSizeAC(const int BLK, const int size) { int Gsize = (size + BLK - 1) / BLK; Gsize = (Gsize > 1024 * 32) ? 1024 * 32 : Gsize; return(Gsize); }