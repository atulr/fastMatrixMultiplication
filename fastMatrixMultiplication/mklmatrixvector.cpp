#include "stdafx.h"
#include "mklmatrixvector.h"


void mklMatrixVector::pushMatrix(MKL_Complex8 *m, int row, int col, int index[][2], int &totalCount) {
	matrices[count] = (MKL_Complex8 *)mkl_malloc( row*col*sizeof( MKL_Complex8 ), 32 );
	if(!matrices[count]) {
		fprintf(stderr, "!!!! mkl matrix memory allocation error. \n");
	}
	matrices[count] = m;
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
