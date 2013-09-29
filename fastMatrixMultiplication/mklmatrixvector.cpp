#include "stdafx.h"
#include "mklmatrixvector.h"

void mklMatrixVector::incrementCount(){
	count++;
}

void mklMatrixVector::pushMatrix(MKL_Complex8 *m, int row, int col) {
	matrices[count++] = (MKL_Complex8 *)mkl_malloc( row*col*sizeof( MKL_Complex8 ), 32 );
	if(!matrices[count]) {
		fprintf(stderr, "!!!! mkl memory allocation error\n");
	}
	matrices[count] = m;
}

void mklMatrixVector::pushVector(MKL_Complex8 *v, int dim) {

}