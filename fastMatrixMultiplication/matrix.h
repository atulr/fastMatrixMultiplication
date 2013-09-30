#include "stdafx.h"
#include "complextype.h"
#include "mklmatrixvector.h"
#include "cudamatrixvector.h"
class Matrix {
	ComplexType *myComplexType; 
	int index[200][2];
	mklMatrixVector mMV;
	cudaMatrixVector cMV;
	int totalCount;
	public:
		int count();
		void incrementCount() {totalCount++;};
		void pushMatrix(ComplexType * matrix, int rows, int cols); //adds the matrix to the respective implementation (cuda or MKL) and creates a memory allocation for a compatible vector on the respective device..
		void multiply(); // will have to figure out the inteface to this call... This is what the interface looks like...
		void updateVector(ComplexType *vector, int index); //this updates the value of the vector at index..
		ComplexType returnVector(); // will return the result of the matrix vector multiplication...
		void totalTimeTaken(); //this needs to have a global DEBUG flag which returns the amortized time complexity...
};