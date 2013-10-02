#include "stdafx.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>
#include <host_defines.h>

class cudaMatrixVector {
	cuComplex *matrices[200];
	cuComplex *vectors[200];
	cuComplex *outVectors[200];
	cuComplex *outVectorsHost[200];
	int matrixDims[200][2];
	int vectorDims[200];
	int count;
	void initializeVector(int col);
public:
	cudaMatrixVector() {count = 0;};
	void updateVector(cuComplex *V, int index);
	void pushMatrix(cuComplex *M, int row, int col, int index[][2], int &totalCount);
	cuComplex* returnVector(int i);
	void multiply(cublasHandle_t handle);
};