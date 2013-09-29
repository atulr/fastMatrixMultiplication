#include "stdafx.h"
#include <mkl.h>
class mklMatrixVector {
	MKL_Complex8 *matrix;
	int count;
	void incrementCount();
public:
	void pushMatrix(MKL_Complex8 *M, int row, int col);
};