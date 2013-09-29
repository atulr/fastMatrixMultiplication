#include "stdafx.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>
#include <host_defines.h>

class cudaMatrixVector {
	cuComplex *matrices[200];
	int count;
	void incrementCount();
public:
	void pushMatrix(cuComplex *M, int row, int col);

};