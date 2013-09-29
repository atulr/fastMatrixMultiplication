#include "matrix.h"
#include "mkl.h"

void Matrix::pushMatrix(ComplexType * matrix, int row, int col) {

	if(row > 3000 || col > 3000) 
		cMV.pushMatrix((cuComplex *) matrix, row, col);
	else 
		mMV.pushMatrix((MKL_Complex8 *) matrix, row, col);
}