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
	index[totalCount][0] = count;
	index[totalCount++][1] = 1;
	initializeVector(col);

}

void cudaMatrixVector::initializeVector(int col) {
	cudaError_t errStatus;
	cublasStatus_t status;
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

}
