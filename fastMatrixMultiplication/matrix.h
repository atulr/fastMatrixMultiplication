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
		Matrix() {myComplexType->complex = 0.0; myComplexType->real = 0.0;};
		int count();
		void incrementCount() {totalCount++;};
		ComplexType* returnComplexType() {return myComplexType;};
		void pushMatrix(ComplexType * matrix, int rows, int cols); //adds the matrix to the respective implementation (cuda or MKL) and creates a memory allocation for a compatible vector on the respective device..
		void multiply(); // will have to figure out the inteface to this call... This is what the interface looks like...
		void updateVector(ComplexType *vector, int index); //this updates the value of the vector at index..
		ComplexType* returnVector(int i); // will return the ith vector of the matrix vector multiplication...
		void totalTimeTaken(); //this needs to have a global DEBUG flag which returns the amortized time complexity...
};