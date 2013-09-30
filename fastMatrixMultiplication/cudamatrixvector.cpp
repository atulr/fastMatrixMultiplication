#include "stdafx.h"
#include "cudaMatrixVector.h"

void cudaMatrixVector::pushMatrix(cuComplex *m, int row, int col, int index[][2], int &totalCount) {
	int matrixSize;
	cudaError_t errStatus;
	cublasStatus_t status;
	matrixSize = row * col;
	matrices[count]  = (cuComplex *)malloc(matrixSize * sizeof(cuComplex));
	if(!matrices[count]) {
		fprintf(stderr, "!!!! malloc allocation error cuda vector matrix \n");
	}
	errStatus = cudaMalloc((void **)&matrices[count], matrixSize * sizeof(matrices[0][0]));
	if(errStatus != cudaSuccess) {
		fprintf(stderr, "!!!! cuda memory allocation error\n");
	}

	status = cublasSetMatrix(row, col, sizeof(cuComplex), m, row, matrices[count], row);
	if(status != cudaSuccess) {
		fprintf(stderr, "!!!! cuda set matrix error\n");
	}
	index[totalCount][0] = count++;
	index[totalCount++][1] = 1;
	initializeVector(col);

}

void cudaMatrixVector::initializeVector(int col) {
	cudaError_t errStatus;
	cublasStatus_t status;

}
