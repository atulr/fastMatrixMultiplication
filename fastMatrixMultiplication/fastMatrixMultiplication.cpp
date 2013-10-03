// fastMatrixMultiplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "matrix.h"

int _tmain(int argc, _TCHAR* argv[])
{
	Matrix m;
	ComplexType *myType;
	myType = (ComplexType*)m.returnComplexType();
	ComplexType *A[200], *B[200], *Av[200], *Bv[200], *Cv[200];
	for(int i = 0; i < 10; i++) {
		A[i] = (ComplexType *)malloc(4096 * 4096 * sizeof(ComplexType));
		Av[i] = (ComplexType *)malloc(4096 * 1 * sizeof(ComplexType));
		Cv[i] = (ComplexType *)malloc(4096 * sizeof(ComplexType));
	}

	for(int i = 0; i < 10; i++) {
		B[i] = (ComplexType *)malloc(2048 * 2048 * sizeof(ComplexType));
		Av[i + 10] = (ComplexType *)malloc(2048 * 1 * sizeof(ComplexType));
	}
	for(int k = 0; k < 10; k++)
		for(int i = 0 ; i < 4096; i++) {
			for(int j = 0; j < 4096 ;j++) {
				A[k][i * 4096 + j].real = (float)(i * 4096 + j);
				A[k][i * 4096 + j].complex = (float)(i * 4096 + j);		
		}
	}

	for(int k = 0; k < 10; k++)
		for(int i = 0 ; i < 2048; i++) {
			for(int j = 0; j < 2048 ;j++) {
			B[k][i * 2048 + j].real = (float)(i * 2048 + j);
			B[k][i * 2048 + j].complex = (float)(i * 2048 + j);
			}
		}
	
	for(int j = 0; j < 10; j++)
		for(int i = 0; i < 4096; i++) {
			Av[j][i].real = (float)i;
			Av[j][i].complex = (float)i;
		}
	
	for(int j = 10; j < 20; j++)
		for(int i = 0; i<2048; i++) {
			Av[j][i].real = (float)i;
			Av[j][i].complex = (float)i;
		}


	for(int i = 0; i < 10; i++)
		m.pushMatrix(A[i], 4096, 4096);
	for(int i = 0; i < 10; i++)
		m.pushMatrix(B[i], 2048, 2048);
	for(int i = 0; i < 20; i++)
		m.updateVector(Av[i], i);
	//m.updateVector(Bv, 1);
	m.multiply();
	//Cv = m.returnVector(0);

	return 0;
}

