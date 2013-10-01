#include "stdafx.h"
#include "matrix.h"
#include "mkl.h"

void Matrix::pushMatrix(ComplexType * matrix, int row, int col) {

	if(row > 3000 || col > 3000) 
		cMV.pushMatrix((cuComplex *) matrix, row, col, index, totalCount);
	else 
		mMV.pushMatrix((MKL_Complex8 *) matrix, row, col, index, totalCount);
}

void Matrix::updateVector(ComplexType *vector, int i) {
	if(index[i][1] == 1) {
		cMV.updateVector((cuComplex *) vector, index[i][0]);
	} else {
		mMV.updateVector((MKL_Complex8 *) vector, index[i][0]);
	}
}

ComplexType* Matrix::returnVector(int i) {
	if(index[i][1] == 1) {
		return (ComplexType*) cMV.returnVector(index[i][0]);
	} else{
		return (ComplexType*) mMV.returnVector(index[i][0]);
	}
}

void Matrix::multiply() {
	cublasHandle_t handle;
	cMV.multiply(handle);
	mMV.multiply();
	cudaError_t cudaStatus = cudaThreadSynchronize();
    if (cudaStatus != cudaSuccess)
    {
         fprintf(stderr, "!!!! GPU program execution error on cudaThreadSynchronize : cudaError=%d,(%s)\n", cudaStatus,cudaGetErrorString(cudaStatus));
         return;
    }

	cublasDestroy(handle);
}
