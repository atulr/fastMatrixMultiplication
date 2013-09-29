#include "stdafx.h"
#include "matrix.h"
#include "mkl.h"

void Matrix::pushMatrix(ComplexType * matrix, int row, int col) {

	if(row > 3000 || col > 3000) 
		cMV.pushMatrix((cuComplex *) matrix, row, col);
	else 
		mMV.pushMatrix((MKL_Complex8 *) matrix, row, col);
}

void Matrix::pushVector(ComplexType * vector, int dim) {
	if(dim > 3000)
		cMV.pushVector((cuComplex *) vector, dim);
	else
		mMV.pushVector((MKL_Complex8 *) vector, dim);
}

void Matrix::multiply(ComplexType *vector, int dim) {

}
