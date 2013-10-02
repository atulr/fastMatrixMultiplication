// fastMatrixMultiplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "matrix.h"

int _tmain(int argc, _TCHAR* argv[])
{
	Matrix m;
	ComplexType *myType;
	myType = (ComplexType*)m.returnComplexType();
	ComplexType *A, *B;
	A = (ComplexType *)malloc(500 * 500 * sizeof(ComplexType));
	B = (ComplexType *)malloc(5000 * 5000 * sizeof(ComplexType));
	for(int i = 0 ; i < 500; i++) {
		for(int j = 0; j < 500 ;j++) {
			A->real = (float)i;
			A->complex = (float)i;
		}
	}

	for(int i = 0 ; i < 5000; i++) {
		for(int j = 0; j < 5000 ;j++) {
			A->real = (float)i;
			A->complex = (float)i;
		}
	}

	m.pushMatrix(A, 500, 500);
	m.pushMatrix(B, 5000, 5000);
	return 0;
}

