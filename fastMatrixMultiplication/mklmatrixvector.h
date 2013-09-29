#include "stdafx.h"
#include <mkl.h>
class mklMatrixVector {
	MKL_Complex8 *matrices[200];
	int count;
	void incrementCount();
public:
	void pushMatrix(MKL_Complex8 *M, int row, int col);
	void pushVector(MKL_Complex8 *V, int dim);
};