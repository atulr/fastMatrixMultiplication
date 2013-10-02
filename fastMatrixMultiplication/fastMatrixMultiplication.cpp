// fastMatrixMultiplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "matrix.h"

int _tmain(int argc, _TCHAR* argv[])
{
	Matrix m;
	ComplexType *myType;
	myType = (ComplexType*)m.returnComplexType();
	ComplexType *A, *B, *Av, *Bv, *Cv;
	A = (ComplexType *)malloc(3 * 3 * sizeof(ComplexType));
	B = (ComplexType *)malloc(3 * 3 * sizeof(ComplexType));
	Av = (ComplexType *)malloc(3 * 1 * sizeof(ComplexType));
	Bv = (ComplexType *)malloc(3 * 1 * sizeof(ComplexType));
	Cv = (ComplexType *)malloc(3 * sizeof(ComplexType));
	for(int i = 0 ; i < 3; i++) {
		for(int j = 0; j < 3 ;j++) {
			A[i * 3 + j].real = (float)(i * 3 + j);
			A[i * 3 + j].complex = (float)(i * 3 + j);		
		}
	}
	for(int i = 0 ; i < 3; i++) {
		for(int j = 0; j < 3 ;j++) {
			B[i * 3 + j].real = (float)(i * 3 + j);
			B[i * 3 + j].complex = (float)(i * 3 + j);
		}
	}

	for(int i = 0; i < 3; i++) {
		Av[i].real = (float)i;
		Av[i].complex = (float)i;
	}

	for(int i = 0; i<3; i++) {
		Bv[i].real = (float)i;
		Bv[i].complex = (float)i;
	}


	m.pushMatrix(A, 3, 3);
	m.pushMatrix(B, 3, 3);
	m.updateVector(Av, 0);
	m.updateVector(Bv, 1);
	m.multiply();
	Cv = m.returnVector(0);
	FILE *fp;

	fp = fopen("d:\\testresult.txt", "w+");
	for(int i = 0; i < 3; i++)
		fprintf(fp, " %f + i %f \n", Cv[i].real, Cv[i].complex);
	fclose(fp);
	return 0;
}

