#ifndef MKLMATRIXVECTOR_H
#define MKLMATRIXVECTOR_H
#include "stdafx.h"
#include <mkl.h>
class mklMatrixVector {
	MKL_Complex8 *matrices[200];
	MKL_Complex8 *vectors[200];
	MKL_Complex8 *outVectors[200];
	int matrixDimensions[200][2];
	int count;
	double totalTime;
	void initializeVector(int col);
public:
	mklMatrixVector() {count = 0;};
	void updateVector(MKL_Complex8 *V, int index);
	void pushMatrix(MKL_Complex8 *M, int row, int col, int index[][2], int &totalCount);
	MKL_Complex8* returnVector(int i);
	void multiply();
	double returnTimeTaken();
};
#endif