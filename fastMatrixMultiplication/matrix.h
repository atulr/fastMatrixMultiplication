#include "stdafx.h"
#include "complextype.h"
#include "mklmatrixvector.h"
#include "cudamatrixvector.h"
class Matrix {
	ComplexType *myComplexType; 
	mklMatrixVector mMV;
	cudaMatrixVector cMV;
	public:
		void pushMatrix(ComplexType * matrix, int rows, int cols); //adds the matrix to the respective implementation (cuda or MKL)
		void multiply(); // will have to figure out the inteface to this call... 
		ComplexType returnVector(); // will return the result of the matrix vector multiplication...
};