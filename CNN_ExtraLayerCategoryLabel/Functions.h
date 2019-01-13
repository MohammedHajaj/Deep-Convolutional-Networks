#pragma once


//==============================================================================================================================================================================================================
// Function declarations for Functions.cpp
//==============================================================================================================================================================================================================


void SaveParameters1(FILE *, float *, float **, float **, float *, float **, int *, int *);

void ReloadParameters1(FILE *, float *, float **, float **, float *, float **, int *, int *);

void SaveParameters2_PlusExtraLayer(FILE *, float *, float **, float **, float **, float **, float *, float *, float *, float *, float **, float **, int *, int *, int, int, int, float);

void ReloadParameters2_PlusExtraLayer(FILE *, float *, float **, float **, float **, float **, float *, float *, float *, float *, float **, float **, int *, int *, int *, int *, int *, float *);

void InitializeTrainingData_PlusCatLabel(unsigned int **, unsigned int **, size_t **, int **, int **, int **, size_t **, size_t **, int *, int *, int);

void InitialzeCuDNN(cudnnTensorDescriptor_t *, cudnnTensorDescriptor_t *, cudnnFilterDescriptor_t *, cudnnConvolutionDescriptor_t *, cudnnTensorDescriptor_t *, cudnnTensorDescriptor_t *, cudnnFilterDescriptor_t *, cudnnConvolutionDescriptor_t *, cudnnConvolutionFwdAlgo_t *, cudnnConvolutionBwdDataAlgo_t *, cudnnConvolutionBwdFilterAlgo_t *, int *, int *, int *);

void AllocateParamGPUMemory(float ***, float ***, float ***, float ***, float ***, float ***, float ***, float ***, float ***, float ***, float ***, float ***, float ***, float **, float **, float **, float **, float **, int ***, bool ***, int *, int *, int *, int *, int *, int *, int *);

void AllocateGPUMemory_ExtraOutputLayer(float **, float **, float **, float **, float **, float **, float **, int *, int *);

void AllocateAuxiliaryGPUMemory(float ***, float **, float **, float **, float **, float **, float **, unsigned int **, float **, float **, int **, int ***, int *, int *, int *, int);

void InitializeConvLayerParam(int *, int *, int *, int *, int *, int *, int *, int *, int *, float **);

void ParameterInitialization(float **, float **, float *, float **, float *, int *, int *);

void ParameterInitialization_ExtraOutputLayer(float *, float *);

void InitializeCudaKernels(dim3 *, dim3 *, dim3 *, dim3 *, dim3 *, dim3 *, dim3 *, dim3 *, int *, int *, int *, int *, int *, int *, int *, int *, int *);

void InitializeCudaKernels_ExtraOutputLayer(dim3 *, int *, int *, int *, int *);

void PrintIterResults(FILE *, float *, float *, int *, int, int, int);

void InitializeMultiCropInference(int **);

void PrintFinalResults(FILE *, float *, float *, int *);

void ReshuffleImages(int *, int *, int);

void FreeTrainSpecificData(float **, float **, float **, float **, float **, float **, float **, float *, float *, float *, float *, float *, float *, int *, int **, int *, unsigned int *, int);

void FreeRemainingMem(unsigned char *, unsigned char *, unsigned char *, float **, float **, float **, float **, float **, float **, float **, int **, bool **, float *, float *, float *, float *, float *, float *, float *, unsigned int *, unsigned int *, size_t *, int *, int *, size_t *, size_t *, float *);

static int GridSizeAC(const int, const int);



















