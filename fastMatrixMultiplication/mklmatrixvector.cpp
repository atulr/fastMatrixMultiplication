#include "stdafx.h"
#include "mklmatrixvector.h"


void mklMatrixVector::pushMatrix(MKL_Complex8 *m, int row, int col, int index[][2], int &totalCount) {
	matrices[count] = (MKL_Complex8 *)mkl_malloc( row*col*sizeof( MKL_Complex8 ), 32 );
	if(!matrices[count]) {
		fprintf(stderr, "!!!! mkl matrix memory allocation error. \n");
	}
	matrices[count] = m;

	matrixDimensions[count][0] = row;
	matrixDimensions[count][1] = col;
	index[totalCount][0] = count;
	index[totalCount++][1] = 0;
	initializeVector(col);
}


void mklMatrixVector::initializeVector(int col) {
	vectors[count] = (MKL_Complex8 *)malloc(col * sizeof(MKL_Complex8));
	if(!vectors[count]) {
		fprintf(stderr, "!!!! mkl in vector memory allocation error. \n");
	}
	outVectors[count] = (MKL_Complex8 *)malloc(col * sizeof(MKL_Complex8));
	if(!outVectors[count]) {
		fprintf(stderr, "!!!! mkl out vector memory allocation error. \n");
	}

	count++;
}

void mklMatrixVector::multiply() {
	MKL_Complex8 alpha, beta;
	alpha.imag = 0.0;
	alpha.real = 1.0;
	beta.imag = 0.0;
	beta.real = 0.0;
	for (int i=0; i<count; i++)
       cblas_cgemv(CblasRowMajor, CblasNoTrans, matrixDimensions[i][0], matrixDimensions[i][1], &alpha, matrices[i], matrixDimensions[i][0], vectors[i], 1, &beta, outVectors[i], 1);

}

void mklMatrixVector::updateVector(MKL_Complex8 *v, int index) {
	vectors[index] = v;
}

MKL_Complex8* mklMatrixVector::returnVector(int i) {
	return outVectors[i];
}
