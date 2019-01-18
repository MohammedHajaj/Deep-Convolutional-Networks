#include"Header.h"

//============================================================================================================================================================
//============================================================================================================================================================
//============================================================================================================================================================
//   This.cu file contains the implementation of core CUDA kernels required to implement a deep feed - forward convolutional network.
//============================================================================================================================================================
//============================================================================================================================================================
//============================================================================================================================================================

//============================================================================================================================================================

template<int>
__global__ void MaxPoolingForward(float *s, float *c, int *Indx, int CRC, int SRC, int Src1, int Src2, int NumCh)
{

	//-------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This cuda kernel implements the forward pass for maxpooling. The input channel size is CRCxCRC, the pooling size is Src1xSrc1,
	the pooling stride is Src2xSrc2, and the output channel size is SRCxSRC where SRC = CRC/Src2. The index of the maximum output
	in each pooling square is stored in the Indx matrix to speed up the backward pass through the maxpooling stage.
	*/

	/**** Argument List ****/
	/*

	NumCh:- Number of output channels per image. Total number of channels in a convolutional layer is NumCh*BatchSize.
	Src1:-  Pooling size is Src1×Src1.
	Src2:-  Pooling stride.
	CRC:-   Input channel size is CRC×CRC (output channel size of previous stage before applying maxpooling).
	SRC:-   Output channel size is SRC×SRC after applying pooling.
	c:-     Input buffer that conatians all input channels (output channels of the previous layer).
	Indx:-  Output buffer to store the positon of the maximum value in each pooling square to be used by MaxPoolingBackward in the backward pass.
	s:-     Output buffer where this cuda kernel stores the all output channels after applying maximum pooling.

	*/
	//-------------------------------------------------------------------------------------------------------------------------------------------------

	int iss = blockIdx.x*blockDim.x + threadIdx.x;
	int SN = SRC*SRC;

	if (iss < SN)
	{
		int n = blockIdx.y + NumCh*blockIdx.z;
		int icx = Src1 * (iss % SRC);
		int icy = Src1 * (iss / SRC);
		int is = iss + n*SN;
		int ic = (n*CRC + icy)*CRC + icx;

		float max = -9.9e+30f;
		int index;
		for (int i = -1; i < Src2 - 1; i++)
			for (int j = -1; j < Src2 - 1; j++)
			{
				int j1 = icx + j, i1 = icy + i, ix = ic + i*CRC + j;
				if (j1 >= 0 && i1 >= 0 && j1 < CRC && i1 < CRC && c[ix] > max) { index = ix;  max = c[index]; }
			}


		s[is] = max;
		Indx[is] = index;
	}
}

//============================================================================================================================================================

template<int>
__global__ void MaxPoolingBackward(float *c, float *s, int *Indx, int SN, int NumCh)
{

	//-------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This cuda kernel implements the backward pass for maxpooling. The only job of this cuda function is to propagate back the error
	signal through the maxpooling stage, there is no parameter update required because the maxpooling stage has no trainable parameters.
	The error signal will only be passed to the location of the maximum value in each pooling square using the Indx matrix which stores
	those maximum values.
	*/

	/**** Argument List ****/
	/*

	NumCh:- Number of output channels per image. Total number of channels in a convolutional layer is NumCh*BatchSize.
	SN:-    The channel size on the output side of the mapooling stage, SN = SRC×SRC.
	s:-     Input buffer that contains the error signal with respect to the activations of all channels on the output side of the maxpooling stage.
	Indx:-  Input buffer where the positon of the maximum value in each pooling square was stored in the forward pass by MaxPoolingForward.
	c:-     Output buffer where this cuda kernel stores the error signal with respect to the activations of all channels on the input side of the maxpooling stage.

	*/
	//-------------------------------------------------------------------------------------------------------------------------------------------------

	int iss = blockIdx.x*blockDim.x + threadIdx.x;
	if (iss < SN)
	{
		int n = blockIdx.y + NumCh*blockIdx.z;
		int is = n*SN + iss;
		atomicAdd(c + Indx[is], s[is]);		
	}
}

//============================================================================================================================================================

template<const int BLOCKSIZE>
__global__ void GlobalAvgPoolingForward(float *S, float *C, int NumCh, int ChSize)
{

	//-------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This cuda kernel implements the forward pass of the global average pooling stage used after the last convolutional layer.
	The input channel size is ChSize which is reduced to a single value that is equal to the average of all values in the channel.
	*/

	/**** Argument List ****/
	/*

	NumCh:-  Number of output channels per image. Total number of channels in a convolutional layer is NumCh*BatchSize.
	ChSize:- Input Channel size.
	c:-      Input buffer that conatians all input channels (output channels of the previous layer).
	s:-      Output buffer where this cuda kernel stores the all output channels after applying global average pooling.

	*/
	//-------------------------------------------------------------------------------------------------------------------------------------------------

	__shared__ float s[BLOCKSIZE];
	int is = threadIdx.x;

	if (is < ChSize)
	{
		int n = blockIdx.x + NumCh*blockIdx.y;
		int b = blockIdx.y;
		int ig1 = n*ChSize + is;
		int ig2 = n + b;

		float sum = 0.0f;
		while (ig1 < (n + 1)*ChSize)
		{
			sum += C[ig1];
			ig1 += BLOCKSIZE;
		}

		s[is] = sum;

		__syncthreads();

		int i = blockDim.x / 2;
		while (i > 0 && is + i < ChSize)
		{
			if (is < i)  s[is] += s[is + i];
			__syncthreads();
			i /= 2;
		}

		if (is == 0) S[ig2] = s[0] / ChSize;
	}
}

//============================================================================================================================================================

template<const int BLOCKSIZE>
__global__ void GlobalAvgPoolingBackward(float *C, float *S, int NumCh, int ChSize)
{

	//-------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This cuda kernel implements the backward pass of the global average pooling stage. This stage has no trainable parameters,
	and therefore this function only propagates back the error signal through the average pooling stage.
	*/

	/**** Argument List ****/
	/*

	NumCh:-  Number of output channels per image. Total number of channels in a convolutional layer is NumCh*BatchSize.
	ChSize:- Input Channel size.
	s:-      Input buffer that contains the error signal with respect to the activations of all channels on the output side of the average pooling stage.
	c:-      Output buffer where this cuda kernel stores the error signal with respect to the activations of all channels on the input side of the average pooling stage.

	*/
	//-------------------------------------------------------------------------------------------------------------------------------------------------

	__shared__ float s_temp;
	int is = threadIdx.x;

	if (is < ChSize)
	{
		int n = blockIdx.x + NumCh*blockIdx.y;
		int b = blockIdx.y;
		int ig1 = n*ChSize + is;
		int ig2 = n + b;

		if (is == 0)s_temp = S[ig2];

		__syncthreads();

		float temp = s_temp / ChSize;
		while (ig1 < (n + 1)*ChSize)
		{
			C[ig1] = temp;
			ig1 += BLOCKSIZE;
		}

	}
}

//============================================================================================================================================================

template < const int SIZE, const int BLOCKSIZE >
__global__ void Softmax(float *y, int *t, int *Indx, float *mse, float *count)
{

	//-------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This cuda kernel implements both the forward and backward passes of the softmax function. Because the softmax stage is the last
	stage in the network, doing the backward propagation of the error signal immediately after doing the forward pass in the same cuda
	function is more efficient and adds minimal cost. The implementation is slightly complex because it takes into consideration the
	possibility of overflow, and the thread block size limitation to 1024 cuda threads per block.
	*/

	/**** Argument List ****/
	/*

	Indx:-  input buffer that stores the indices or locations of the images in the current batch.
	t:-     input buffer that contains the image labels.
	count:- output variable to store the total number of images that were correctly classified for the training set or validation set.
	mse:-   output variable to accumulate the mean square error of the training set or validation set
	y:-     input/output buffer where the inputs to the softmax function are stored and where this kernel stores the error signal at the input side of the softmax stage.

	*/
	//-------------------------------------------------------------------------------------------------------------------------------------------------

	__shared__ float s[BLOCKSIZE];
	__shared__ double sD[BLOCKSIZE];
	__shared__ int indx[BLOCKSIZE], s_t;

	int is = threadIdx.x;
	int n = blockIdx.x;
	int ig = n*SIZE + is, ig1 = ig;
	if (is == 0) { int indx1 = Indx[n]; s_t = t[indx1]; }

	float tempy[8], max = -9.9e+30f, mse1;
	double tempyD[8], sum;
	int k = 0, ix;

	while (ig < (n + 1)*SIZE)
	{
		tempy[k] = y[ig];
		if (tempy[k]>max){ max = tempy[k]; ix = k; }
		k++;
		ig += BLOCKSIZE;
	}

	s[is] = max;
	indx[is] = is + BLOCKSIZE*ix;

	__syncthreads();

	int i = BLOCKSIZE / 2;
	while (i > 0)
	{
		if (is < i && is + i < SIZE)
		{
			if (s[is + i] > s[is])   { s[is] = s[is + i];   indx[is] = indx[is + i]; }
		}
		__syncthreads();
		i /= 2;
	}

	max = s[0];

	//----------------------------------------------------------------------------------------------

	ig = ig1;
	sum = 0.0; k = 0;

	float a = 1.0;
	while (ig < (n + 1)*SIZE)
	{
		if (max > 700){ a = (700 / max); tempy[k] *= a; }
		tempyD[k] = exp(double(tempy[k]));
		sum += tempyD[k];
		k++;
		ig += BLOCKSIZE;
	}
	sD[is] = sum;

	__syncthreads();

	i = BLOCKSIZE / 2;
	while (i > 0)
	{
		if (is < i && is + i < SIZE)  sD[is] += sD[is + i];
		__syncthreads();
		i /= 2;
	}

	sum = sD[0];

	//----------------------------------------------------------------------------------------------

	ig = ig1;
	mse1 = 0.0f;
	k = 0;

	while (ig < (n + 1)*SIZE)
	{
		float temp = float(tempyD[k] / (sum + 2.0e-20));
		if (is + k*BLOCKSIZE == s_t) temp -= 1.0f;
		y[ig] = a*temp;
		mse1 += temp * temp;
		k++;
		ig += BLOCKSIZE;
	}

	s[is] = mse1;

	__syncthreads();

	i = BLOCKSIZE / 2;
	while (i > 0)
	{
		if (is < i && is + i < SIZE)  s[is] += s[is + i];
		__syncthreads();
		i /= 2;
	}

	if (is == 0)
	{
		mse[n] += s[0];
		if (indx[0] == s_t) atomicAdd(count, 1.0f);
	}
}

//============================================================================================================================================================

template < const int SIZE, const int BLOCKSIZE >
__global__ void SoftmaxInference(float *ys, float *y, int *t, float *mse)
{

	//-------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This cuda kernel implements the softmax function used in the inference stage. This function is similar in structure to the Softmax
	cuda function used in the training phase. However, rather than propagating back the error signal, this function calculates and stores
	the whole output label for each test image in Ys which then can be analyzed to calculate the mean square error, the confusion matrix,
	and the classification rate of the test set.
	*/

	/**** Argument List ****/
	/*

	t:-     input buffer that contains the image labels.
	mse:-   output variable to accumulate the mean square error of the test set.
	y:-     input buffer where the inputs to the softmax function are stored.
	ys:-    output buffer where this kernel stores the whole predicted labels of the test images.

	*/
	//-------------------------------------------------------------------------------------------------------------------------------------------------

	__shared__ float s[BLOCKSIZE];
	__shared__ double sD[BLOCKSIZE];
	__shared__ int s_t;

	int is = threadIdx.x;
	int n = blockIdx.x;
	int ig = n*SIZE + is, ig1 = ig;

	if (is == 0){ s_t = t[n]; }

	float tempy[8], max = -9.9e+30f, mse1;
	double tempyD[8], sum;
	int k = 0;

	while (ig < (n + 1)*SIZE)
	{
		tempy[k] = y[ig];
		max = fmaxf(max, tempy[k]);
		k++;
		ig += BLOCKSIZE;
	}

	s[is] = max;

	__syncthreads();

	int i = BLOCKSIZE / 2;
	while (i > 0)
	{
		if (is < i && is + i < SIZE)
		{
			s[is] = fmaxf(s[is], s[is + i]);
		}
		__syncthreads();
		i /= 2;
	}

	max = s[0];

	//----------------------------------------------------------------------------------------------

	ig = ig1;
	sum = 0.0; k = 0;

	while (ig < (n + 1)*SIZE)
	{
		if (max > 700)tempy[k] *= 700 / max;
		tempyD[k] = exp(double(tempy[k]));
		sum += tempyD[k];
		k++;
		ig += BLOCKSIZE;
	}
	sD[is] = sum;

	__syncthreads();

	i = BLOCKSIZE / 2;
	while (i > 0)
	{
		if (is < i && is + i < SIZE)  sD[is] += sD[is + i];
		__syncthreads();
		i /= 2;
	}

	sum = sD[0];

	//----------------------------------------------------------------------------------------------

	ig = ig1;
	mse1 = 0.0f;
	k = 0;

	while (ig < (n + 1)*SIZE)
	{
		float temp = float(tempyD[k] / (sum + 2.0e-20));
		ys[ig] += temp;
		if (is + k*BLOCKSIZE == s_t) temp -= 1.0f;
		mse1 += temp * temp;
		k++;
		ig += BLOCKSIZE;
	}

	s[is] = mse1;

	__syncthreads();

	i = BLOCKSIZE / 2;
	while (i > 0)
	{
		if (is < i && is + i < SIZE)  s[is] += s[is + i];
		__syncthreads();
		i /= 2;
	}

	if (is == 0) { mse[n] += s[0]; }

}

//============================================================================================================================================================

template < const int SIZE, const int BLOCKSIZE >
__global__ void SoftmaxInference2(float *ys, float *y, int *t, int *Indx, float *mse)
{
	//-------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This cuda kernel is similar to the SoftmaxInference() kernel with the addition of the Indx buffer which is generated earlier by the ReshuffleImagesMT()
	function to store the indices of batches of images from all tasks in a round robin order. 
	*/
	//-------------------------------------------------------------------------------------------------------------------------------------------------

	__shared__ float s[BLOCKSIZE];
	__shared__ double sD[BLOCKSIZE];
	__shared__ int s_t, s_indx1;

	int is = threadIdx.x;
	int n = blockIdx.x;
	int ig = n*SIZE + is, ig1 = ig;

	if (is == 0) { int indx1 = Indx[n]; s_indx1 = indx1; s_t = t[indx1]; }

	__syncthreads();

	int ig2 = s_indx1*SIZE + is;

	float tempy[8], max = -9.9e+30f, mse1;
	double tempyD[8], sum;
	int k = 0;

	while (ig < (n + 1)*SIZE)
	{
		tempy[k] = y[ig];
		max = fmaxf(max, tempy[k]);
		k++;
		ig += BLOCKSIZE;
	}

	s[is] = max;

	__syncthreads();

	int i = BLOCKSIZE / 2;
	while (i > 0)
	{
		if (is < i && is + i < SIZE)
		{
			s[is] = fmaxf(s[is], s[is + i]);
		}
		__syncthreads();
		i /= 2;
	}

	max = s[0];

	//----------------------------------------------------------------------------------------------

	ig = ig1;
	sum = 0.0; k = 0;

	while (ig < (n + 1)*SIZE)
	{
		if (max > 700)tempy[k] *= 700 / max;
		tempyD[k] = exp(double(tempy[k]));
		sum += tempyD[k];
		k++;
		ig += BLOCKSIZE;
	}
	sD[is] = sum;

	__syncthreads();

	i = BLOCKSIZE / 2;
	while (i > 0)
	{
		if (is < i && is + i < SIZE)  sD[is] += sD[is + i];
		__syncthreads();
		i /= 2;
	}

	sum = sD[0];

	//----------------------------------------------------------------------------------------------

	ig = ig1;
	mse1 = 0.0f;
	k = 0;

	while (ig < (n + 1)*SIZE)
	{
		float temp = float(tempyD[k] / (sum + 2.0e-20));
		ys[ig2] += temp;
		if (is + k*BLOCKSIZE == s_t) temp -= 1.0f;
		mse1 += temp * temp;
		k++;
		ig += BLOCKSIZE;
		ig2 += BLOCKSIZE;
	}

	s[is] = mse1;

	__syncthreads();

	i = BLOCKSIZE / 2;
	while (i > 0)
	{
		if (is < i && is + i < SIZE)  s[is] += s[is + i];
		__syncthreads();
		i /= 2;
	}

	if (is == 0) { mse[n] += s[0]; }

}


//============================================================================================================================================================

template < const int SIZE, const int BLOCKSIZE >
__global__ void LogSigmoid(float *y, int *t, int *Indx, float *mse, float *count)
{

	//-------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This cuda kernel implements the forward and backward passes of the log sigmoid function 1/(1+exp(-x)).
	*/

	/**** Argument List ****/
	/*

	Indx:-  input buffer that stores the indices or locations of the images in the current batch.
	t:-     input buffer that contains the image labels.
	count:- output variable to store the total number of images that were correctly classified for the training set or validation set.
	mse:-   output variable to accumulate the mean square error of the training set or validation set
	y:-     input/output buffer where the inputs to the log sigmoid function are stored and where this kernel stores the error signal at the input side of the softmax stage.

	*/
	//-------------------------------------------------------------------------------------------------------------------------------------------------

	__shared__ float s[BLOCKSIZE];
	__shared__ int indx[BLOCKSIZE], s_t;

	int is = threadIdx.x;
	int n = blockIdx.x;
	int ig = n*SIZE + is, ig1 = ig;

	if (is == 0) { int indx1 = Indx[n]; s_t = t[indx1]; }

	float tempy[8], max = -9.9e+30f, mse1;
	int k = 0, ix;

	while (ig < (n + 1)*SIZE)
	{
		tempy[k] = y[ig];
		if (tempy[k]>max){ max = tempy[k]; ix = k; }
		k++;
		ig += BLOCKSIZE;
	}

	s[is] = max;
	indx[is] = is + BLOCKSIZE*ix;

	__syncthreads();

	int i = BLOCKSIZE / 2;
	while (i > 0)
	{
		if (is < i && is + i < SIZE)
		{
			if (s[is + i] > s[is])   { s[is] = s[is + i];   indx[is] = indx[is + i]; }
		}
		__syncthreads();
		i /= 2;
	}

	//----------------------------------------------------------------------------------------------

	ig = ig1;
	mse1 = 0.0f;
	k = 0;

	while (ig < (n + 1)*SIZE)
	{
		float temp = float(1.0f / (1.0f + expf(-tempy[k])));
		if (is + k*BLOCKSIZE == s_t) temp -= 1.0f;
		y[ig] = temp;
		mse1 += temp * temp;
		k++;
		ig += BLOCKSIZE;
	}

	s[is] = mse1;

	__syncthreads();

	i = BLOCKSIZE / 2;
	while (i > 0)
	{
		if (is < i && is + i < SIZE)  s[is] += s[is + i];
		__syncthreads();
		i /= 2;
	}

	if (is == 0)
	{
		mse[n] += s[0];
		if (indx[0] == s_t) atomicAdd(count, 1.0f);
	}
}

//============================================================================================================================================================

template < const int SIZE, const int BLOCKSIZE >
__global__ void LogSigmoidInference(float *ys, float *y)
{

	//-------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This cuda kernel implements the log sigmoid function 1/(1+exp(-x)) used in the Inference stage. It stores the whole output label of each test image in ys.
	*/

	/**** Argument List ****/
	/*

	y:-     input buffer where the inputs to the softmax function are stored.
	ys:-    output buffer where this kernel stores the whole predicted labels of the test images.

	*/
	//-------------------------------------------------------------------------------------------------------------------------------------------------

	int is = threadIdx.x;
	int n = blockIdx.x;
	int ig = n*SIZE + is;

	while (ig < (n + 1)*SIZE)
	{
		float temp = float(1.0f / (1.0f + expf(-y[ig])));
		ys[ig] += temp;
		ig += BLOCKSIZE;
	}

}

//============================================================================================================================================================

template < int>
__global__ void Add_Mtx(float *c, float *a, int SIZE)
{

	//-------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This cuda kernel is an auxiliary cuda function that adds two GPU matrices c = c + a.
	*/

	/**** Argument List ****/
	/*

	a:-     input matrix.
	c:-    input/output matrix  to store c = c + a..
	SIZE:- size of the input/output matrices.

	*/
	//-------------------------------------------------------------------------------------------------------------------------------------------------

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int stride = blockDim.x*gridDim.x;
	while (i < SIZE)
	{
		c[i] += a[i];
		i += stride;
	}
}

//============================================================================================================================================================

template < int>
__global__ void Update_SGD_WDecay(float *c, float *a, float lr, float lmda, int SIZE)
{
	//-------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This cuda kernel updates the parameters in matrix w with the derivatives in matrix dw. The upate equation implements steepest
	gradient decent with L2 regularization (weight decay).
	*/

	/**** Argument List ****/
	/*

	lr:-     learning rate.
	lmda:-   weight decay parameter.
	dw:-     input buffer that contains the derivatives.
	w:-      input/output buffer to store the updated values of the trainable parameters in matrix w.
	SIZE:-   size of the input/output matrices.

	*/
	//-------------------------------------------------------------------------------------------------------------------------------------------------

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int stride = blockDim.x*gridDim.x;
	while (i < SIZE)
	{
		c[i] = (1 - lr*lmda)*c[i] - lr * a[i];
		i += stride;
	}
}

//============================================================================================================================================================

template < int>
__global__ void Update_RMSprop1(float *w, float *v, float *dw, float lr, float lmda, int SIZE, int iter)
{
	//-------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This cuda kernel updates the parameters in matrix w with the current derivatives in matrix dw and the running averages of the
	derivatives in matrix v. The update equation implements Root Mean Square Propagation (RMSprop) with L2 regularization (weight decay).
	The initialization of the running average of the derivatives based on the time step (iteration number) is borrowed from the Adam algorithm.
	*/

	/**** Argument List ****/
	/*

	lr:-     learning rate.
	lmda:-   weight decay parameter.
	iter:-   current training iteration.
	dw:-     input buffer that contains the derivatives.
	v:-      input/output buffer that maintains the running average of the squared derivative per parameter.
	w:-      input/output buffer to store the updated values of the trainable parameters in matrix w.
	SIZE:-   size of the input/output matrices.

	*/
	//-------------------------------------------------------------------------------------------------------------------------------------------------

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int stride = blockDim.x*gridDim.x;
	while (i < SIZE)
	{
		float gamma = 0.999f;
		v[i] = gamma*v[i] + (1 - gamma)*dw[i] * dw[i];
		float m = v[i] / (1 - powf(gamma, float(iter)));
		w[i] = (1 - lr*lmda)*w[i] - lr* dw[i] / (sqrtf(m) + 0.00000001);
		i += stride;
	}
}

//============================================================================================================================================================

template < int>
__global__ void Update_RMSprop2(float *A, float *V, float *DA, float lr, float lmda, int SIZE, int iter)
{
	//-------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	Update_RMSprop2 is similar to Update_RMSprop1, but it is used for smaller matrices.
	*/

	/**** Argument List ****/
	/*

	lr:-     learning rate.
	lmda:-   weight decay parameter.
	iter:-   current training iteration.
	dA:-     input buffer that contains the derivatives.
	V:-      input/output buffer that maintains the running average of the squared derivative per parameter.
	A:-      input/output buffer to store the updated values of the trainable parameters in matrix w.
	SIZE:-   size of the input/output matrices.

	*/
	//-------------------------------------------------------------------------------------------------------------------------------------------------

	int i = threadIdx.x + blockIdx.x*blockDim.x;

	if (i < SIZE)
	{
		float gamma = 0.999f;
		V[i] = gamma*V[i] + (1 - gamma)*DA[i] * DA[i];
		float m = V[i] / (1 - powf(gamma, float(iter)));
		A[i] = (1 - lr*lmda)*A[i] - lr* DA[i] / (sqrtf(m) + 0.00000001);
	}
}

//============================================================================================================================================================

template<int>
__global__ void DataAugmentation(float *XCrop, unsigned char *Red, unsigned char *Green, unsigned char *Blue, unsigned int * Height, unsigned int *Width, size_t *Start, int *Indx, unsigned int *Crop, float *RGB, float *Cropf)
{
	//-------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This cuda kernel implements data augmentations which is applied at the start of each training iteration. The augmentation is applied
	to a batch of images where the Red matrix stores the Red channels for all images, Green matrix stores the Green channels for all images,
	and the Blue matrix stores the Blue channels for all images. The Height and Width matrices store the height and width of each image in
	the batch, and the Start matrix stores the starting address of each image in the Red, Green, and Blue buffers. The Indx matrix stores the
	indices of the images in the current batch selected by the reshuffle algorithm. This function crops a random rectangular with size randomly
	selected to be between 8% and 100% of the image size and with aspect ratio randomly selected to be between 3/4 and 4/3. Then the cropped
	rectangular is fitted to the square window size of the network. Random horizontal flipping and color augmentation is also added.  Bilinear
	interpolation is used for scaling.
	*/

	/**** Argument List ****/
	/*

	Height:- input buffer that holds the height of all images in the batch.
	Width:-  input buffer that holds the width of all images in the batch.
	Start:-  input buffer that holds the starting position of all images in the batch.
	Red:-    input buffer where the red input channels for all images in the batch are stored.
	Green:-  input buffer where the green input channels for all images in the batch are stored.
	Blue:-   input buffer where the blue input channels for all images in the batch are stored.
	Indx:-   input buffer that contains the indices of the images in the current batch selected by the reshuffle algorithm.
	Crop:-   input buffer that contains integer random values used to choose the cropping position for each image, and decide on horizontal flipping.
	RGB:-    input buffer that contains 3 random values per image each added to one of the RGB channels for colour augmentation.
	Cropf:-  input buffer that contains 2 random values per image, one decides the amount of scaling, and the other decides the amount of change to the aspect ratio.
	XCrop:-  output buffer to store a batch of data augmented images.

	*/
	//-------------------------------------------------------------------------------------------------------------------------------------------------

	__shared__ unsigned int s_crop[3], s_height, s_width, s_indx;
	__shared__ float s_RGB[3], s_cropf[2];
	__shared__ size_t s_start;

	int is = threadIdx.x;
	int ii = is + blockIdx.x*blockDim.x;
	int n = blockIdx.y;

	if (is < 3)
	{
		s_crop[is] = Crop[3 * n + is];
		s_RGB[is] = RGB[3 * n + is];
		if (is < 2) s_cropf[is] = Cropf[2 * n + is];
	}

	if (ii < ISize1)
	{
		int ix = ii % IR1;
		int iy = ii / IR1;

		if (is == 0)
		{
			s_indx = Indx[n];
			s_height = Height[s_indx];
			s_width = Width[s_indx];
			s_start = Start[s_indx];
		}

		__syncthreads();

		int H = s_height;
		int W = s_width;
		int Hc, Wc;

		size_t start = s_start;

		float a = 0.08f + s_cropf[0] * (1.0f - 0.08f); //float a = 0.1914 + s_cropf[0] * (0.765625 - 0.1914);//
		float minHW = fminf(float(H), float(W));
		float smax = fminf(1.3333333f, (W*W) / (minHW*minHW*a));
		float smin = fmaxf(0.75f, (minHW*minHW*a) / (H*H));
		float s = smin + s_cropf[1] * (smax - smin);
		Wc = int(minHW*sqrtf(a*s));
		Hc = int(minHW*sqrtf(a / s));

		float ScaleH = float(IR1 - 1) / float(Hc - 1);
		float ScaleW = float(IR1 - 1) / float(Wc - 1);



		int xd = s_crop[0] % (H + 1 - Hc);
		int yd = s_crop[1] % (W + 1 - Wc);
		int flip = s_crop[2] % 10;

		int ic = ix + IR1*iy + 3 * n*ISize1;

		float ixs, iys;
		int ixs1, ixs2, iys1, iys2;

		ixs = float(ix) / ScaleH + float(xd);
		if (flip < 5) iys = float(iy) / ScaleW + float(yd); else iys = float(IR1 - 1 - iy) / ScaleW + float(yd);

		ixs1 = floorf(ixs);
		ixs2 = ceilf(ixs);
		iys1 = floorf(iys);
		iys2 = ceilf(iys);

		if (iys1 < iys2)
		{
			if (ixs1 < ixs2)
			{
				int ia1 = ixs1 + iys1*H + start;
				int ia2 = ixs2 + iys1*H + start;
				int ia3 = ixs1 + iys2*H + start;
				int ia4 = ixs2 + iys2*H + start;

				float t1 = Red[ia1] * (ixs2 - ixs) + Red[ia2] * (ixs - ixs1);
				float t2 = Red[ia3] * (ixs2 - ixs) + Red[ia4] * (ixs - ixs1);
				XCrop[ic] = (s_RGB[0] + t1*(iys2 - iys) + t2*(iys - iys1)) / 255.0f - 0.5f;

				t1 = Green[ia1] * (ixs2 - ixs) + Green[ia2] * (ixs - ixs1);
				t2 = Green[ia3] * (ixs2 - ixs) + Green[ia4] * (ixs - ixs1);
				XCrop[ic + ISize1] = (s_RGB[1] + t1*(iys2 - iys) + t2*(iys - iys1)) / 255.0f - 0.5f;

				t1 = Blue[ia1] * (ixs2 - ixs) + Blue[ia2] * (ixs - ixs1);
				t2 = Blue[ia3] * (ixs2 - ixs) + Blue[ia4] * (ixs - ixs1);
				XCrop[ic + 2 * ISize1] = (s_RGB[2] + t1*(iys2 - iys) + t2*(iys - iys1)) / 255.0f - 0.5f;
			}
			else
			{
				int ia1 = ixs1 + iys1*H + start;
				int ia2 = ixs1 + iys2*H + start;

				XCrop[ic] = (s_RGB[0] + Red[ia1] * (iys2 - iys) + Red[ia2] * (iys - iys1)) / 255.0f - 0.5f;
				XCrop[ic + ISize1] = (s_RGB[1] + Green[ia1] * (iys2 - iys) + Green[ia2] * (iys - iys1)) / 255.0f - 0.5f;;
				XCrop[ic + 2 * ISize1] = (s_RGB[2] + Blue[ia1] * (iys2 - iys) + Blue[ia2] * (iys - iys1)) / 255.0f - 0.5f;

			}

		}
		else
		{
			if (ixs1 < ixs2)
			{
				int ia1 = ixs1 + iys1*H + start;
				int ia2 = ixs2 + iys1*H + start;

				XCrop[ic] = (s_RGB[0] + Red[ia1] * (ixs2 - ixs) + Red[ia2] * (ixs - ixs1)) / 255.0f - 0.5f;
				XCrop[ic + ISize1] = (s_RGB[1] + Green[ia1] * (ixs2 - ixs) + Green[ia2] * (ixs - ixs1)) / 255.0f - 0.5f;
				XCrop[ic + 2 * ISize1] = (s_RGB[2] + Blue[ia1] * (ixs2 - ixs) + Blue[ia2] * (ixs - ixs1)) / 255.0f - 0.5f;
			}
			else
			{
				int ia1 = ixs1 + iys1*H + start;

				XCrop[ic] = (s_RGB[0] + Red[ia1]) / 255.0f - 0.5f;
				XCrop[ic + ISize1] = (s_RGB[1] + Green[ia1]) / 255.0f - 0.5f;
				XCrop[ic + 2 * ISize1] = (s_RGB[2] + Blue[ia1]) / 255.0f - 0.5f;
			}

		}

	}

}

//============================================================================================================================================================

template<const int EpochT>
__global__ void DataAugmentationInference(float *XCrop, unsigned char *Red, unsigned char *Green, unsigned char *Blue, unsigned int * Height, unsigned int *Width, size_t *Start, int *MTX, unsigned int *Flip, int epoch)
{
	//-------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This cuda kernel implements data augmentations for test images in the inference stage. This function can do Single-Crop and Multi-Crop
	inference based on the EpochTs value which contains the number of crops per image. If EpochTs is equal to 1 then this function will do
	a single crop-crop inference. If EpochTs >1 this function will do a multi-crop inference. The cropping locations and scales for each test
	image are stored in the MTX matrix. Each crop will be horizontally flipped with 0.5 probability. The Prediction for each test image is
	equal to the average predictions of all the crops stored in the MTX matrix. EpochTs is a control variable stored in "ControlVariables.h".
	Bilinear interpolation is used for scaling.
	*/

	/**** Argument List ****/
	/*

	Height:- input buffer that holds the height of all images in the batch.
	Width:-  input buffer that holds the width of all images in the batch.
	Start:-  input buffer that holds the starting position of all images in the batch.
	Red:-    input buffer where the red input channels for all images in the batch are stored.
	Green:-  input buffer where the green input channels for all images in the batch are stored.
	Blue:-   input buffer where the blue input channels for all images in the batch are stored.
	MTX:-    input buffer that contains the cropping positions and amount of scaling applied to all images in the batch.
	Flip:-   input buffer that contains one random value per image that is used to decide on horizontal flipping.
	epoch:-  represents the crop number in multi-crop inference.
	XCrop:-  output buffer to store a test batch of augmented images.

	*/
	//-------------------------------------------------------------------------------------------------------------------------------------------------

	__shared__ unsigned int  s_height, s_width, s_flip, s_mtx[3];
	__shared__ size_t s_start;

	int is = threadIdx.x;
	int ii = is + blockIdx.x*blockDim.x;
	int n = blockIdx.y;
	if (is < 3) s_mtx[is] = MTX[n * 3 * EpochT + 3 * epoch + is];

	if (ii < ISize1)
	{
		int ix = ii % IR1;
		int iy = ii / IR1;

		if (is == 0)
		{
			s_height = Height[n];
			s_width = Width[n];
			s_start = Start[n];
			s_flip = Flip[n];
		}

		__syncthreads();

		int H = s_height;
		int W = s_width;
		int Hs, Ws;
		float Hf = H, Wf = W;
		size_t start = s_start;

		if (H > W)
		{
			Ws = s_mtx[0];
			Hs = Ws*(Hf / Wf);
		}
		else
		{
			Hs = s_mtx[0];
			Ws = Hs*(Wf / Hf);
		}

		int xd = s_mtx[1];
		int flip = s_flip % 10;
		int yd = s_mtx[2];

		int ic = ix + IR1*iy + 3 * n*ISize1;

		float ixs, iys;
		int ixs1, ixs2, iys1, iys2;

		ixs = (ix + xd)*((Hf - 1) / (Hs - 1));
		if (flip < 5) iys = (iy + yd)*((Wf - 1) / (Ws - 1)); else iys = (IR1 - 1 - iy + yd)*((Wf - 1) / (Ws - 1)); //else iys = (Ws - 1 - iy - yd)*((Wf - 1) / (Ws - 1));

		ixs1 = floorf(ixs);
		ixs2 = ceilf(ixs);
		iys1 = floorf(iys);
		iys2 = ceilf(iys);

		if (iys1 < iys2)
		{
			if (ixs1 < ixs2)
			{
				int ia1 = ixs1 + iys1*H + start;
				int ia2 = ixs2 + iys1*H + start;
				int ia3 = ixs1 + iys2*H + start;
				int ia4 = ixs2 + iys2*H + start;

				float t1 = Red[ia1] * (ixs2 - ixs) + Red[ia2] * (ixs - ixs1);
				float t2 = Red[ia3] * (ixs2 - ixs) + Red[ia4] * (ixs - ixs1);
				XCrop[ic] = (t1*(iys2 - iys) + t2*(iys - iys1)) / 255.0f - 0.5f;

				t1 = Green[ia1] * (ixs2 - ixs) + Green[ia2] * (ixs - ixs1);
				t2 = Green[ia3] * (ixs2 - ixs) + Green[ia4] * (ixs - ixs1);
				XCrop[ic + ISize1] = (t1*(iys2 - iys) + t2*(iys - iys1)) / 255.0f - 0.5f;

				t1 = Blue[ia1] * (ixs2 - ixs) + Blue[ia2] * (ixs - ixs1);
				t2 = Blue[ia3] * (ixs2 - ixs) + Blue[ia4] * (ixs - ixs1);
				XCrop[ic + 2 * ISize1] = (t1*(iys2 - iys) + t2*(iys - iys1)) / 255.0f - 0.5f;
			}
			else
			{
				int ia1 = ixs1 + iys1*H + start;
				int ia2 = ixs1 + iys2*H + start;

				XCrop[ic] = (Red[ia1] * (iys2 - iys) + Red[ia2] * (iys - iys1)) / 255.0f - 0.5f;
				XCrop[ic + ISize1] = (Green[ia1] * (iys2 - iys) + Green[ia2] * (iys - iys1)) / 255.0f - 0.5f;;
				XCrop[ic + 2 * ISize1] = (Blue[ia1] * (iys2 - iys) + Blue[ia2] * (iys - iys1)) / 255.0f - 0.5f;

			}

		}
		else
		{
			if (ixs1 < ixs2)
			{
				int ia1 = ixs1 + iys1*H + start;
				int ia2 = ixs2 + iys1*H + start;

				XCrop[ic] = (Red[ia1] * (ixs2 - ixs) + Red[ia2] * (ixs - ixs1)) / 255.0f - 0.5f;
				XCrop[ic + ISize1] = (Green[ia1] * (ixs2 - ixs) + Green[ia2] * (ixs - ixs1)) / 255.0f - 0.5f;
				XCrop[ic + 2 * ISize1] = (Blue[ia1] * (ixs2 - ixs) + Blue[ia2] * (ixs - ixs1)) / 255.0f - 0.5f;
			}
			else
			{
				int ia1 = ixs1 + iys1*H + start;

				XCrop[ic] = Red[ia1] / 255.0f - 0.5f;
				XCrop[ic + ISize1] = Green[ia1] / 255.0f - 0.5f;
				XCrop[ic + 2 * ISize1] = Blue[ia1] / 255.0f - 0.5f;
			}

		}

	}

}


//============================================================================================================================================================

template<const int EpochT>
__global__ void DataAugmentationInference2(float *XCrop, unsigned char *Red, unsigned char *Green, unsigned char *Blue, unsigned int * Height, unsigned int *Width, size_t *Start, int *Indx, int *MTX, unsigned int *Flip, int epoch)
{
	//-------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This cuda kernel is similar to the DataAugmentationInference() kernel with the addition of the Indx buffer which is generated earlier by the ReshuffleImagesMT()
	function to store the indices of batches of images from all tasks in a round robin order. 
	*/
	//-------------------------------------------------------------------------------------------------------------------------------------------------

	__shared__ unsigned int  s_height, s_width, s_flip, s_mtx[3], s_indx;
	__shared__ size_t s_start;

	int is = threadIdx.x;
	int ii = is + blockIdx.x*blockDim.x;
	int n = blockIdx.y;

	if (is == 0)
	{
		s_indx = Indx[n];
		s_height = Height[s_indx];
		s_width = Width[s_indx];
		s_start = Start[s_indx];
		s_flip = Flip[n];
	}

	__syncthreads();

	if (is < 3) s_mtx[is] = MTX[s_indx * 3 * EpochT + 3 * epoch + is];

	__syncthreads();

	if (ii < ISize1)
	{
		int ix = ii % IR1;
		int iy = ii / IR1;

		int H = s_height;
		int W = s_width;
		int Hs, Ws;
		float Hf = H, Wf = W;
		size_t start = s_start;

		if (H > W)
		{
			Ws = s_mtx[0];
			Hs = Ws*(Hf / Wf);
		}
		else
		{
			Hs = s_mtx[0];
			Ws = Hs*(Wf / Hf);
		}

		int xd = s_mtx[1];
		int flip = s_flip % 10;
		int yd = s_mtx[2];

		int ic = ix + IR1*iy + 3 * n*ISize1;

		float ixs, iys;
		int ixs1, ixs2, iys1, iys2;

		ixs = (ix + xd)*((Hf - 1) / (Hs - 1));
		if (flip < 5) iys = (iy + yd)*((Wf - 1) / (Ws - 1)); else iys = (IR1 - 1 - iy + yd)*((Wf - 1) / (Ws - 1)); //else iys = (Ws - 1 - iy - yd)*((Wf - 1) / (Ws - 1));

		ixs1 = floorf(ixs);
		ixs2 = ceilf(ixs);
		iys1 = floorf(iys);
		iys2 = ceilf(iys);

		if (iys1 < iys2)
		{
			if (ixs1 < ixs2)
			{
				int ia1 = ixs1 + iys1*H + start;
				int ia2 = ixs2 + iys1*H + start;
				int ia3 = ixs1 + iys2*H + start;
				int ia4 = ixs2 + iys2*H + start;

				float t1 = Red[ia1] * (ixs2 - ixs) + Red[ia2] * (ixs - ixs1);
				float t2 = Red[ia3] * (ixs2 - ixs) + Red[ia4] * (ixs - ixs1);
				XCrop[ic] = (t1*(iys2 - iys) + t2*(iys - iys1)) / 255.0f - 0.5f;

				t1 = Green[ia1] * (ixs2 - ixs) + Green[ia2] * (ixs - ixs1);
				t2 = Green[ia3] * (ixs2 - ixs) + Green[ia4] * (ixs - ixs1);
				XCrop[ic + ISize1] = (t1*(iys2 - iys) + t2*(iys - iys1)) / 255.0f - 0.5f;

				t1 = Blue[ia1] * (ixs2 - ixs) + Blue[ia2] * (ixs - ixs1);
				t2 = Blue[ia3] * (ixs2 - ixs) + Blue[ia4] * (ixs - ixs1);
				XCrop[ic + 2 * ISize1] = (t1*(iys2 - iys) + t2*(iys - iys1)) / 255.0f - 0.5f;
			}
			else
			{
				int ia1 = ixs1 + iys1*H + start;
				int ia2 = ixs1 + iys2*H + start;

				XCrop[ic] = (Red[ia1] * (iys2 - iys) + Red[ia2] * (iys - iys1)) / 255.0f - 0.5f;
				XCrop[ic + ISize1] = (Green[ia1] * (iys2 - iys) + Green[ia2] * (iys - iys1)) / 255.0f - 0.5f;;
				XCrop[ic + 2 * ISize1] = (Blue[ia1] * (iys2 - iys) + Blue[ia2] * (iys - iys1)) / 255.0f - 0.5f;

			}

		}
		else
		{
			if (ixs1 < ixs2)
			{
				int ia1 = ixs1 + iys1*H + start;
				int ia2 = ixs2 + iys1*H + start;

				XCrop[ic] = (Red[ia1] * (ixs2 - ixs) + Red[ia2] * (ixs - ixs1)) / 255.0f - 0.5f;
				XCrop[ic + ISize1] = (Green[ia1] * (ixs2 - ixs) + Green[ia2] * (ixs - ixs1)) / 255.0f - 0.5f;
				XCrop[ic + 2 * ISize1] = (Blue[ia1] * (ixs2 - ixs) + Blue[ia2] * (ixs - ixs1)) / 255.0f - 0.5f;
			}
			else
			{
				int ia1 = ixs1 + iys1*H + start;

				XCrop[ic] = Red[ia1] / 255.0f - 0.5f;
				XCrop[ic + ISize1] = Green[ia1] / 255.0f - 0.5f;
				XCrop[ic + 2 * ISize1] = Blue[ia1] / 255.0f - 0.5f;
			}

		}

	}

}

//============================================================================================================================================================

template<int>
__global__ void DataAugmentationValidate(float *XCrop, unsigned char *Red, unsigned char *Green, unsigned char *Blue, unsigned int * Height, unsigned int *Width, size_t *Start, int *Indx)
{
	//-------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This cuda kernel is a simplified version of DataAugmentation used with the validation images. A single central crop with size equal
	to 224/256 of the maximum square size in the image is used to calculate the validation error. Bilinear interpolation is used for scaling.
	*/

	/**** Argument List ****/
	/*

	Height:- input buffer that holds the height of all images in the batch.
	Width:-  input buffer that holds the width of all images in the batch.
	Start:-  input buffer that holds the starting position of all images in the batch.
	Red:-    input buffer where the red input channels for all images in the batch are stored.
	Green:-  input buffer where the green input channels for all images in the batch are stored.
	Blue:-   input buffer where the blue input channels for all images in the batch are stored.
	Indx:-   input buffer that contains the indices of the images in the current batch selected by the reshuffle algorithm.
	XCrop:-  output buffer to store a batch of data augmented images.

	*/
	//-------------------------------------------------------------------------------------------------------------------------------------------------

	__shared__ unsigned int s_height, s_width, s_indx;
	__shared__ size_t s_start;

	int is = threadIdx.x;
	int ii = is + blockIdx.x*blockDim.x;

	if (ii < ISize1)
	{
		int ix = ii % IR1;
		int iy = ii / IR1;

		int n = blockIdx.y;

		if (is == 0)
		{
			s_indx = Indx[n];
			s_height = Height[s_indx];
			s_width = Width[s_indx];
			s_start = Start[s_indx];
		}

		__syncthreads();

		int H = s_height;
		int W = s_width;
		int Hs, Ws;
		float Hf = H, Wf = W;
		size_t start = s_start;

		if (H > W)
		{
			Ws = IR1;//1.143f*IR1; //
			Hs = Ws*(Hf / Wf);
		}
		else
		{
			Hs = IR1;//1.143f*IR1; //
			Ws = Hs*(Wf / Hf);
		}

		int xd = (Hs - IR1) / 2;
		int yd = (Ws - IR1) / 2;

		int ic = ix + IR1*iy + 3 * n*ISize1;

		float ixs, iys;
		int ixs1, ixs2, iys1, iys2;

		ixs = (ix + xd)*((Hf - 1) / (Hs - 1));
		iys = (iy + yd)*((Wf - 1) / (Ws - 1));
		//if (flip == 0) iys = (iy + yd)*((Wf - 1) / (Ws - 1)); else iys = (IR1 - 1 - iy + yd)*((Wf - 1) / (Ws - 1)); //else iys = (Ws - 1 - iy - yd)*((Wf - 1) / (Ws - 1));

		ixs1 = floorf(ixs);
		ixs2 = ceilf(ixs);
		iys1 = floorf(iys);
		iys2 = ceilf(iys);

		if (iys1 < iys2)
		{
			if (ixs1 < ixs2)
			{
				int ia1 = ixs1 + iys1*H + start;
				int ia2 = ixs2 + iys1*H + start;
				int ia3 = ixs1 + iys2*H + start;
				int ia4 = ixs2 + iys2*H + start;

				float t1 = Red[ia1] * (ixs2 - ixs) + Red[ia2] * (ixs - ixs1);
				float t2 = Red[ia3] * (ixs2 - ixs) + Red[ia4] * (ixs - ixs1);
				XCrop[ic] = (t1*(iys2 - iys) + t2*(iys - iys1)) / 255.0f - 0.5f;

				t1 = Green[ia1] * (ixs2 - ixs) + Green[ia2] * (ixs - ixs1);
				t2 = Green[ia3] * (ixs2 - ixs) + Green[ia4] * (ixs - ixs1);
				XCrop[ic + ISize1] = (t1*(iys2 - iys) + t2*(iys - iys1)) / 255.0f - 0.5f;

				t1 = Blue[ia1] * (ixs2 - ixs) + Blue[ia2] * (ixs - ixs1);
				t2 = Blue[ia3] * (ixs2 - ixs) + Blue[ia4] * (ixs - ixs1);
				XCrop[ic + 2 * ISize1] = (t1*(iys2 - iys) + t2*(iys - iys1)) / 255.0f - 0.5f;
			}
			else
			{
				int ia1 = ixs1 + iys1*H + start;
				int ia2 = ixs1 + iys2*H + start;

				XCrop[ic] = (Red[ia1] * (iys2 - iys) + Red[ia2] * (iys - iys1)) / 255.0f - 0.5f;
				XCrop[ic + ISize1] = (Green[ia1] * (iys2 - iys) + Green[ia2] * (iys - iys1)) / 255.0f - 0.5f;;
				XCrop[ic + 2 * ISize1] = (Blue[ia1] * (iys2 - iys) + Blue[ia2] * (iys - iys1)) / 255.0f - 0.5f;

			}

		}
		else
		{
			if (ixs1 < ixs2)
			{
				int ia1 = ixs1 + iys1*H + start;
				int ia2 = ixs2 + iys1*H + start;

				XCrop[ic] = (Red[ia1] * (ixs2 - ixs) + Red[ia2] * (ixs - ixs1)) / 255.0f - 0.5f;
				XCrop[ic + ISize1] = (Green[ia1] * (ixs2 - ixs) + Green[ia2] * (ixs - ixs1)) / 255.0f - 0.5f;
				XCrop[ic + 2 * ISize1] = (Blue[ia1] * (ixs2 - ixs) + Blue[ia2] * (ixs - ixs1)) / 255.0f - 0.5f;
			}
			else
			{
				int ia1 = ixs1 + iys1*H + start;

				XCrop[ic] = Red[ia1] / 255.0f - 0.5f;
				XCrop[ic + ISize1] = Green[ia1] / 255.0f - 0.5f;
				XCrop[ic + 2 * ISize1] = Blue[ia1] / 255.0f - 0.5f;
			}

		}

	}

}

//============================================================================================================================================================

template<int BLOCKSIZE>
__global__ void BatchNormForward1a(float *SMU, float *X, int NumCh, int ChSize)
{
	//-------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This cuda kernel calcualtes the mean and variance per thread block.
	*/

	/**** Argument list****/
	/*

	NumCh:-   Number of output channels per image. Total number of channels in a convolutional layer is NumCh*BatchSize.
	ChSize:-  The size of each output channel.
	X :-      input buffer that contains the activations of all output channels before applying BN.
	SMU:-     output buffer where this function stores all means and variances calculated per thread block.

	*/
	//-------------------------------------------------------------------------------------------------------------------------------------------------

	__shared__ float s1[BLOCKSIZE], s2[BLOCKSIZE];
	int is = threadIdx.x;
	int n = blockIdx.x + NumCh*blockIdx.y;
	int ig = is + n*ChSize;
	float temp, sum = 0, sum_sq = 0;

	while (ig < (n + 1)*ChSize)
	{
		temp = X[ig];
		sum += temp;
		sum_sq += temp*temp;
		ig += BLOCKSIZE;
	}

	s1[is] = sum;
	s2[is] = sum_sq;
	__syncthreads();

	int i = BLOCKSIZE / 2;
	while (i > 0)
	{
		if (is < i)  { s1[is] += s1[is + i]; s2[is] += s2[is + i]; }
		__syncthreads();
		i /= 2;
	}

	if (is == 0)
	{
		SMU[2 * n] = s1[0];
		SMU[2 * n + 1] = s2[0];
	}
}

//============================================================================================================================================================

template<int BLOCKSIZE>
__global__ void BatchNormForward1b(float *SMU, int Ch, int TotalChSize)
{
	//-------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This cuda kernel accumulates the means and variances per thread block calculated by BatchNormForward1a to calculate the mean and variance
	per output channel. The reason for this two stage calculation of the means and variances is caused by the layout of the output channels in
	the GPU memory used by the convolutional layer implementation of the cudnn.lib library. The layout is CNHW where the order of the tensor
	inner dimensions is Width, Height, N for image index and Channel. If the layout was NCHW the calculations of the means and variances can
	easily and efficiently be implemented in a single stage. Anyway splitting the calculation into two consecutive stages adds minimal overhead.
	*/

	/**** Argument list****/
	/*

	NumCh:-        Number of output channels per image. Total number of channels in a convolutional layer is NumCh*BatchSize.
	TotalChSize:-  The size of each output channel across all images in the batch TotalChSize = ChSize*BatchSize.
	SMU:-          Output buffer where this function calculates and stores a total of NumCh mean-variance pairs for each of the NumCh output channels.

	*/
	//-------------------------------------------------------------------------------------------------------------------------------------------------

	int is = blockIdx.x*blockDim.x + threadIdx.x;

	if (is < Ch)
	{

		int ix = 2 * is;
		int size = 2 * Ch*BatchSize;
		float sum = 0.0f, sum_sq = 0.0f;

		while (ix < size)
		{
			sum += SMU[ix];
			sum_sq += SMU[ix + 1];
			ix += 2 * Ch;
		}

		float temp = TotalChSize;
		sum /= temp;
		SMU[2 * is] = sum;
		temp = sum_sq / temp - sum*sum;
		SMU[2 * is + 1] = sqrtf(temp + 0.0001);
	}
}

//============================================================================================================================================================

template<int BLOCKSIZE>
__global__ void BatchNormForward2(float *Y, float *X, float *SMU, float *Param, int NumCh, int ChSize)
{
	//-------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This cuda kernel uses the means and variances calculated by BatchNormForward1a and BatchNormForward1b to apply batch normalization to the
	output channels.
	*/

	/**** Argument list***/
	/*

	NumCh:-   Number of output channels per image. Total number of channels in a convolutional layer is NumCh*BatchSize
	ChSize:-  The size of each output channel.
	Param:-   A buffer that contains the BN trainable parameters beta and gamma. There is a total of NumCh beta-gamma pairs, one for each of the NumCh output channels.
	X :-      input buffer that contains the activations of all output channels before applying BN.
	SMU:-     input buffer that contains a means-variance pair per output channel, and each pair will be used to normalize the activations of the corresponding output channel.
	Y :-      output buffer where this function stores the normalized activations of all output channels.

	*/
	//-------------------------------------------------------------------------------------------------------------------------------------------------

	__shared__ float s_param[2], s_smu[2];
	int is = threadIdx.x;
	int ix = blockIdx.x*BLOCKSIZE + is;
	int n = blockIdx.y;
	int b = blockIdx.z;

	if (is < 2)
	{
		s_param[is] = Param[2 * n + is];
		s_smu[is] = SMU[2 * n + is];
	}
	__syncthreads();

	if (ix < ChSize)
	{
		int ig = (NumCh*b + n)*ChSize + ix;
		float temp = (X[ig] - s_smu[0]) / s_smu[1];
		temp = s_param[0] * temp + s_param[1];
		Y[ig] = fmaxf(temp, 0);
	}
}

//============================================================================================================================================================

template<int BLOCKSIZE>
__global__ void BatchNormBackward2(float *DParam, float *Derv, float *Param, float *SMU, float *DY, float *X, int NumCh, int ChSize)
{
	//-------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This cuda kernel is the first stage that propagates the error signal back through the batch normalization stage. DY contains the error
	signal at the output side of the BN stage. This kernel calculates the derivatives for the BN trainable parameters in DParam, and partially
	propagates the error signal back to the inputs of the BN stage and stores these intermediate values in Derv.
	*/

	/****Argument list****/
	/*

	NumCh:-   Number of output channels per image. Total number of channels in a convolutional layer is NumCh*BatchSize
	ChSize:-  The size of each output channel.
	Param:-   input buffer that contains the BN trainable parameters beta and gamma. There is a total of NumCh beta-gamma pairs, one for each of the NumCh output channels.
	X :-      input buffer that contains the activations of all output channels before applying BN.
	SMU:-     input buffer that contains a means-variance pair per output channel, and each pair will be used to normalize the activations of the corresponding output channel.
	DY :-     input buffer that contains the error signal at the outputs of the BN stage. The derivatives of the lost function with respect to the outputs of BN.
	Derv:-    output buffer where this function calculates and stores a total of NumCh pairs of intermediate values that will used by the next stage to propagate back the error signal to the inputs of the BN stage.
	DParam:-  output buffer where this function calculates and stores the derivatives of beta and gamma, the trainable parameters of BN.

	*/
	//-------------------------------------------------------------------------------------------------------------------------------------------------

	__shared__ float s1[BLOCKSIZE], s2[BLOCKSIZE], s3[BLOCKSIZE];
	__shared__ float s_smu[2], s_param[2];

	int is = threadIdx.x;
	int n = blockIdx.x;
	int b = blockIdx.y;
	int n2 = n + NumCh*b;

	if (is < 2)
	{
		s_smu[is] = SMU[2 * n + is];
		s_param[is] = Param[2 * n + is];
	}
	__syncthreads();

	float mu = s_smu[0], inv_sigma = 1.0f / s_smu[1], gamma = s_param[0], beta = s_param[1];

	int ig = is + n2*ChSize;
	float temp1, temp2, temp, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
	float a = gamma*inv_sigma;
	float b1 = beta - a*mu;

	while (ig < (n2 + 1)*ChSize)
	{
		temp2 = X[ig];
		temp = a*temp2 + b1;
		if (temp>0)
		{
			temp1 = DY[ig];
			sum1 += temp1;
			sum2 += temp1*(temp2 - mu) * inv_sigma;
			sum3 += temp1*(temp2 - mu);
		}
		else
		{
			DY[ig] = 0;
		}

		ig += BLOCKSIZE;
	}

	s1[is] = sum1;
	s2[is] = sum2;
	s3[is] = sum3;
	__syncthreads();

	int i = BLOCKSIZE / 2;
	while (i > 0)
	{
		if (is < i)  { s1[is] += s1[is + i]; s2[is] += s2[is + i]; s3[is] += s3[is + i]; }
		__syncthreads();
		i /= 2;
	}

	if (is == 0)
	{
		atomicAdd(DParam + 2 * n, s2[0]);
		atomicAdd(DParam + 2 * n + 1, s1[0]);
		atomicAdd(Derv + 2 * n, gamma*s1[0]);
		atomicAdd(Derv + 2 * n + 1, gamma*s3[0]);
	}
}

//============================================================================================================================================================

template<int BLOCKSIZE>
__global__ void BatchNormBackward1(float *X, float *DY, float *Param, float *SMU, float *Derv, int NumCh, int ChSize)
{
	//-------------------------------------------------------------------------------------------------------------------------------------------------

	/*
	This cuda kernel completes the back propagation of the error signal through the BN stage.
	*/

	/****Argument list****/
	/*

	NumCh:-  Number of output channels per image. Total number of channels in a convolutional layer is NumCh*BatchSize
	ChSize:- The size of each output channel.
	Param:-  input buffer that contains the BN trainable parameters beta and gamma. There is a total of NumCh beta-gamma pairs, one for each of the NumCh output channels.
	SMU:-    input buffer that contains a means-variance pair per output channel, and each pair will be used to normalize the activations of the corresponding output channel.
	DY :-    input buffer that contains the error signal at the outputs of the BN stage. The derivatives of the lost function with respect to the outputs of BN.
	Derv:-   input buffer which contains a total of NumCh pairs of intermediate values that will used by the this function to propagate back ther error signal to the inputs (X) of the BN stage.
	X :-     output buffer where this function calculates and stores the error signal with respect to the inputs of the BN stage.

	*/
	//-------------------------------------------------------------------------------------------------------------------------------------------------

	__shared__ float s_smu[2], s_derv[2], s_gamma;
	int is = threadIdx.x;
	int ix = blockIdx.x*BLOCKSIZE + is;
	int n = blockIdx.y;
	int b = blockIdx.z;

	if (is < 2)
	{
		s_smu[is] = SMU[2 * n + is];
		s_derv[is] = Derv[2 * n + is];
		if (is == 0) { s_gamma = Param[2 * n]; }
	}
	__syncthreads();

	if (ix < ChSize)
	{
		float temp;
		float mu = s_smu[0], inv_sigma = 1.0f / s_smu[1];
		float derv1 = s_derv[0], derv2 = s_derv[1], inv_m = 1.0f / (BatchSize*ChSize);

		int ig = (NumCh*b + n)*ChSize + ix;
		temp = inv_sigma*(s_gamma*DY[ig] - derv1*inv_m - (X[ig] - mu)*derv2*inv_m*inv_sigma*inv_sigma);
		X[ig] = temp;
	}

}

//============================================================================================================================================================

template<int BLOCKSIZE>
__global__ void BatchNormForward22(float *Y, float *X, float *Y0, bool *F, float *SMU, float *Param, int NumCh, int ChSize)
{
	//-------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	BatchNormForward22 is similar to BatchNormForward2, but it has an additional input Y0, which is an input from a residual connection. Also this
	kernel stores the sign of the output in F to be used in the backward pass. Therefore, when the current stage (Layer) has an additional input
	coming from a previous stage through a residual connection, BatchNormForward22 is used instead of BatchNormForward2.
	*/

	/****Argument list****/
	/*

	NumCh :-  Number of output channels per image. Total number of channels in a convolutional layer is NumCh*BatchSize
	ChSize :- The size of each output channel.
	Param :-  A buffer that contains the BN trainable parameters beta and gamma. There is a total of NumCh beta-gamma pairs, one for each of the NumCh output channels.
	X :-      input buffer that contains the activations of all output channels before applying BN.
	Y0 :-     input buffer that contains the activations of the jump-ahead residual connections.
	SMU :-    input buffer that contains a means-variance pair per output channel, and each pair will be used to normalize the activations of the corresponding output channel.
	F :-      output buffer that holds the signs of each output element in Y0 which will be used in the backward pass to propagate the error signal through the ReLUs.
	Y :-      output buffer where this function stores the normalized activations of all output channels.

	*/
	//-------------------------------------------------------------------------------------------------------------------------------------------------

	__shared__ float s_param[2], s_smu[2];
	int is = threadIdx.x;
	int ix = blockIdx.x*BLOCKSIZE + is;
	int n = blockIdx.y;
	int b = blockIdx.z;

	if (is < 2)
	{
		s_param[is] = Param[2 * n + is];
		s_smu[is] = SMU[2 * n + is];
	}
	__syncthreads();

	if (ix < ChSize)
	{
		int ig = (NumCh*b + n)*ChSize + ix;
		float temp = (X[ig] - s_smu[0]) / s_smu[1];
		temp = s_param[0] * temp + s_param[1] + Y0[ig];
		temp = fmaxf(temp, 0);
		Y[ig] = temp;
		F[ig] = (temp>0) ? 1 : 0;

	}
}

//============================================================================================================================================================

template<int BLOCKSIZE>
__global__ void BatchNormBackward22(float *DParam, float *Derv, float *Param, float *SMU, float *DY, bool *F, float *X, int NumCh, int ChSize)
{
	//-------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	BatchNormBackward22 is similar to BatchNormBackward2, but it has an additional input F, which is the sign of the BN output in forward pass.
	Therefore, when the current stage (Layer) has an additional input coming from a previous stage through a residual connection, BatchNormBackward22
	is used instead of BatchNormBackward2.
	*/

	/**** Argument list****/
	/*

	NumCh:-  Number of output channels per image. Total number of channels in a convolutional layer is NumCh*BatchSize
	ChSize:- The size of each output channel.
	Param:-  input buffer that contains the BN trainable parameters beta and gamma. There is a total of NumCh beta-gamma pairs, one for each of the NumCh output channels.
	X :-     input buffer that contains the activations of all output channels before applying BN.
	F :-     input buffer that is used to propagate the error signal back through the ReLU activation function.
	SMU:-    input buffer that contains a means-variance pair per output channel, and each pair will be used to normalize the activations of the corresponding output channel.
	DY :-    input buffer that contains the error signal at the outputs of the BN stage. The derivatives of the lost function with respect to the outputs of BN.
	Derv:-   output buffer where this function calculates and stores a total of NumCh pairs of intermediate values that will used by the next stage to propagate back the error signal to the inputs of the BN stage.
	DParam:- output buffer where this function calculates and stores the derivatives of beta and gamma, the trainable parameters of BN.

	*/
	//-------------------------------------------------------------------------------------------------------------------------------------------------

	__shared__ float s1[BLOCKSIZE], s2[BLOCKSIZE], s3[BLOCKSIZE];
	__shared__ float s_smu[2], s_param[2];

	int is = threadIdx.x;
	int n = blockIdx.x;
	int b = blockIdx.y;
	int n2 = n + NumCh*b;

	if (is < 2)
	{
		s_smu[is] = SMU[2 * n + is];
		s_param[is] = Param[2 * n + is];
	}
	__syncthreads();

	float mu = s_smu[0], inv_sigma = 1.0f / s_smu[1], gamma = s_param[0];

	int ig = is + n2*ChSize;
	float temp1, temp2, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;


	while (ig < (n2 + 1)*ChSize)
	{
		temp2 = X[ig];
		if (F[ig]>0)
		{
			temp1 = DY[ig];
			sum1 += temp1;
			sum2 += temp1*(temp2 - mu) * inv_sigma;
			sum3 += temp1*(temp2 - mu);
		}
		else
		{
			DY[ig] = 0;
		}

		ig += BLOCKSIZE;
	}

	s1[is] = sum1;
	s2[is] = sum2;
	s3[is] = sum3;
	__syncthreads();

	int i = BLOCKSIZE / 2;
	while (i > 0)
	{
		if (is < i)  { s1[is] += s1[is + i]; s2[is] += s2[is + i]; s3[is] += s3[is + i]; }
		__syncthreads();
		i /= 2;
	}

	if (is == 0)
	{
		atomicAdd(DParam + 2 * n, s2[0]);
		atomicAdd(DParam + 2 * n + 1, s1[0]);
		atomicAdd(Derv + 2 * n, gamma*s1[0]);
		atomicAdd(Derv + 2 * n + 1, gamma*s3[0]);
	}
}

//============================================================================================================================================================

template<int BLOCKSIZE>
__global__ void BatchNormForwardT1b(float *SMU, float *SMUs, int NumCh, int count)
{
	//-------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	BatchNormForwardT1b is similar to BatchNormForward1b, but it has an extra output SMUs to accumulate the means and variances from all
	training images. This kernel will only be executed after the last training epoch. After training stops these accumulated values will
	averaged by AdjustFixedMeansStds.
	*/

	/**** Argument list****/
	/*

	NumCh:-       Number of output channels per image. Total number of channels in a convolutional layer is NumCh*BatchSize
	TotalChSize:- The size of each output channel across all images in the batch TotalChSize = ChSize*BatchSize.
	SMU:-         input buffer that contains a means-variance pair per output channel, and each pair will be used to normalize the activations of the corresponding output channel.
	SMUs:-        Output buffer where this function calculates and stores a total of NumCh fixed mean-variance pairs that will be used in the inference stage.

	*/

	//-------------------------------------------------------------------------------------------------------------------------------------------------

	int is = blockIdx.x*blockDim.x + threadIdx.x;

	if (is < NumCh)
	{

		int ix = 2 * is;
		int size = 2 * NumCh*BatchSize;
		float sum = 0.0f, sum_sq = 0.0f;

		while (ix < size)
		{
			sum += SMU[ix];
			sum_sq += SMU[ix + 1];
			ix += 2 * NumCh;
		}

		float temp = count;
		sum /= temp;
		SMU[2 * is] = sum;
		SMUs[2 * is] += sum;
		temp = sum_sq / temp - sum*sum;
		SMU[2 * is + 1] = sqrtf(temp + 0.0001);
		SMUs[2 * is + 1] += temp;
	}
}

//============================================================================================================================================================

template<int>
__global__ void AdjustFixedMeansStds(float *SMU, int NumCh, int TrainSizeM)
{
	//-------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This cuda kernel uses the accumulated values of means and variances calculated by BatchNormForward1b, to calculated the fixed means and
	variances that will be used in the inference stage.
	*/

	/**** Argument list****/
	/*

	NumCh:-       Number of output channels.
	SMU:-         input buffer that conatins the accumulated means and variances for all training data.

	*/

	//-------------------------------------------------------------------------------------------------------------------------------------------------

	int ig = threadIdx.x + blockIdx.x*blockDim.x;
	float temp = float(TrainSizeM / (NumTasks*BatchSize));
	if (ig < NumCh)
	{
		float temp_value = SMU[ig] / temp;

		if (ig % 2 == 1)
		{
			temp_value = sqrtf(temp_value + 0.0001);
		}
		SMU[ig] = temp_value;
	}
}

//============================================================================================================================================================

template<int>
__global__ void RGBrandPCA(float *RGBrand, float *rand1, int SIZE)
{
	//-------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	This cuda kernel implements calculates a set of 3 stochastic values per image to be added to the 3 RGB channels for the purpose of colour augmentation.
	For each random variable in the input buffer rand1 this kernel will calculate a corresponding stochastic value in RGBrand based on PCA analysis of the
	RGB pixel values of all the training set.
	*/

	/****Argument List****/
	/*

	rand1:- input buffer of random values drawn from a normal distribution with zero mean and unity variance.
	RGBrand:- output buffer to store the

	*/
	//-------------------------------------------------------------------------------------------------------------------------------------------------

	int is = threadIdx.x;
	int ig = is + blockIdx.x*blockDim.x;
	if (ig < SIZE)
	{
		float alpha1 = rand1[3 * ig] * 6.9514;
		float alpha2 = rand1[3 * ig + 1] * 17.3739;
		float alpha3 = rand1[3 * ig + 2] * 305.65817;

		float vr1 = -0.4000, vr2 = -0.7061, vr3 = 0.58426;
		float vg1 = 0.80526, vg2 = 0.0336, vg3 = 0.59196;
		float vb1 = -0.4376, vb2 = 0.7073, vb3 = 0.55517;

		RGBrand[3 * ig] = vr1*alpha1 + vr2*alpha2 + vr3*alpha3;
		RGBrand[3 * ig + 1] = vg1*alpha1 + vg2*alpha2 + vg3*alpha3;
		RGBrand[3 * ig + 2] = vb1*alpha1 + vb2*alpha2 + vb3*alpha3;
	}
}

//============================================================================================================================================================







