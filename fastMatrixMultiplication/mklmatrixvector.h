#include "stdafx.h"
#include <mkl.h>
class mklMatrixVector {
	MKL_Complex8 *matrices[200];
	MKL_Complex8 *vectors[200];
	MKL_Complex8 *outVectors[200];
	int count;
	void initializeVector(int col);
public:
	void updateVector(MKL_Complex8 *V, int index);
	void pushMatrix(MKL_Complex8 *M, int row, int col, int index[][2], int &totalCount);
};