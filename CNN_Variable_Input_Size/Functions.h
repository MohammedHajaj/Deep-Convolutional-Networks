#pragma once


//==============================================================================================================================================================================================================
// Function declarations for Functions.cpp
//==============================================================================================================================================================================================================


void SaveParameters1(FILE *, float *, float **, float **, float *, float **, int *, int *, int);

void ReloadParameters1(FILE *, float *, float **, float **, float *, float **, int *, int *, int);

void SaveParameters2(FILE *, float *, float **, float **, float **, float **, float *, float *, float **, float **, int *, int *, int, int, int, int, float);

void ReloadParameters2(FILE *, float *, float **, float **, float **, float **, float *, float *, float **, float **, int *, int *, int, int *, int *, int *, float *);

void InitializeTrainingData(unsigned int **, unsigned int **, size_t **, int **, int **, size_t **, size_t **, int *, int *, int);

void InitialzeCuDNN_Var(cudnnTensorDescriptor_t[NumWin][CL], cudnnTensorDescriptor_t[NumWin][CL], cudnnFilterDescriptor_t *, cudnnConvolutionDescriptor_t *, cudnnTensorDescriptor_t[NumWin][CL], cudnnTensorDescriptor_t[NumWin][CL], cudnnFilterDescriptor_t *, cudnnConvolutionDescriptor_t *, cudnnConvolutionFwdAlgo_t *, cudnnConvolutionBwdDataAlgo_t *, cudnnConvolutionBwdFilterAlgo_t *, int *, Var_Param *);

void AllocateParamGPUMemory_Varm(float ***, float ***, float ***, float ***, float ***, float ***, float ***, float ***, float ***, float ***, float ***, float ***, float ***, float **, float **, float **, float **, float **, int ***, bool ***F, int *, int *, Var_Param *, int *, int *);

void AllocateAuxiliaryGPUMemory_Varm(float ***, float **, float **, float **, float **, float **, float **, unsigned int **, float **, float **, int **, int ***, int *, Var_Param *, int *, int);

void InitializeConvLayerParam_Var(Var_Param *, int *, int *, int *, int *, float **);

void ParameterInitialization(float **, float **, float *, float **, float *, int *, int *);

void InitializeCudaKernels_Var(Var_gridSizes *, dim3 *, dim3 *, dim3 *, dim3 *, dim3 *, int *, int *, int *, Var_Param *, int *, int *, int *);

void PrintIterResults(FILE *, float *, float *, int *, int, int, int);

void InitializeMultiCropInference(int **);

void PrintFinalResults(FILE *, float *, float *, int *);

void ReshuffleImages(int *, int *, int);

void FreeTrainSpecificData(float **, float **, float **, float **, float **, float **, float **, float *, float *, float *, float *, float *, float *, int *, int **, int *, unsigned int *, int);

void FreeRemainingMem(unsigned char *, unsigned char *, unsigned char *, float **, float **, float **, float **, float **, float **, float **, int **, bool **, float *, float *, float *, float *, float *, float *, float *, unsigned int *, unsigned int *, size_t *, int *, int *, size_t *, size_t *, float *);

static int GridSizeAC(const int, const int);



















