#include "stdafx.h"
#include "cudaMatrixVector.h"

void cudaMatrixVector::pushMatrix(cuComplex *m, int row, int col, int index[][2], int &totalCount) {
	int matrixSize;
	cudaError_t errStatus;
	cublasStatus_t status;
	matrixSize = row * col;
	matrices[count]  = (cuComplex *)malloc(matrixSize * sizeof(cuComplex));
	if(!matrices[count]) {
		fprintf(stderr, "!!!! cuda matrix malloc allocation error. \n");
	}
	errStatus = cudaMalloc((void **)&matrices[count], matrixSize * sizeof(matrices[0][0]));
	if(errStatus != cudaSuccess) {
		fprintf(stderr, "!!!! cuda cudaMalloc matrix allocation error. \n");
	}

	status = cublasSetMatrix(row, col, sizeof(cuComplex), m, row, matrices[count], row);

	if(status != cudaSuccess) {
		fprintf(stderr, "!!!! cuda set matrix error\n");
	}
	matrixDims[count][0] = row;
	matrixDims[count][1] = col;
	index[totalCount][0] = count;
	index[totalCount++][1] = 1;
	
	initializeVector(col);

}

void cudaMatrixVector::initializeVector(int col) {
	cudaError_t errStatus;
	vectors[count] = (cuComplex *)malloc(col * sizeof(cuComplex));
	if(!vectors[count]) {
		fprintf(stderr, "!!!! cuda in vector malloc error\n");
	}
	outVectors[count] = (cuComplex *)malloc(col * sizeof(cuComplex)); 
	if(!outVectors[count]) {
		fprintf(stderr, "!!!! cuda out vector malloc error\n");
	}

	errStatus = cudaMalloc((void **)&vectors[count], col * sizeof(cuComplex));
	if(errStatus != cudaSuccess) {
		 fprintf(stderr, "!!!! cuda in vector cudaMalloc error\n");
	}

	errStatus = cudaMalloc((void **)&outVectors[count], col * sizeof(cuComplex));
	if(errStatus != cudaSuccess) {
		 fprintf(stderr, "!!!! cuda out vector cudaMalloc error\n");
	}
	vectorDims[count] = col;
	count++;

}

void cudaMatrixVector::updateVector(cuComplex *V, int index) {
	cublasStatus_t status;

	status = cublasSetVector(vectorDims[index], sizeof(cuComplex), V, 1, vectors[index], 1);

	if(status != cudaSuccess) {
		fprintf(stderr, "!!!! cuda update vector error\n");
	}
}

cuComplex* cudaMatrixVector::returnVector(int i) {
	return outVectors[i];
}

void cudaMatrixVector::multiply(cublasHandle_t handle) {
	cublasStatus_t status;
	cuComplex alph, bet;
	alph.x = 1.0;
	alph.y = 0.0;
	bet.x = 0.0;
	bet.y = 0.0;

	for(int i = 0; i < count; i++) {
		status = cublasCgemv(handle, CUBLAS_OP_T, matrixDims[i][0], matrixDims[i][1],
                                      &alph, matrices[i], matrixDims[i][0],
									  vectors[i] , 1, &bet, outVectors[i], 1);
		if(status != cudaSuccess) {
			fprintf(stderr, "!!!! cublasCgemv error .. \n");
			return;
		}
	}
}