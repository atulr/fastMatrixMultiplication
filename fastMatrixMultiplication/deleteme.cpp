
//// matrixmultiplication.cpp : Defines the entry point for the console application.
////
//
//#include "stdafx.h"
//// CUDA runtime
//#include <cuda_runtime.h>
//#include <cublas_v2.h>
//#include <ctime>
//// CUDA and CUBLAS functions
//#include <helper_functions.h>
//
//#ifndef min
//#define min(a,b) ((a < b) ? a : b)
//#endif
//#ifndef max
//#define max(a,b) ((a > b) ? a : b)
//#endif
//
//
//void multiply(const float *A, const float *B, float *C, const int m) {
//    // Do the actual multiplication
// //   cublasSgemv(handle, CUBLAS_OP_N, m, m, alpha, A, m, B, 1, beta, C, 1);
//
//	//cublasDestroy(handle);
//}
//int _tmain(int argc, _TCHAR* argv[])
//{
//	
//	
//	printf("[Matrix Multiply CUBLAS] - Starting...\n");
//	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
//	nr_rows_A = nr_cols_A = nr_rows_B = nr_rows_C = 5000;
//	nr_cols_B = nr_cols_C = 1;
//	int numberOfIterations = 100;
//	float timetaken;
//	cudaEvent_t start, stop;
//	const float alf = 1;
//    const float bet = 0;
//    const float *alpha = &alf;
//    const float *beta = &bet;
//	const int numberOfMultiplications = 70;
//	cublasHandle_t handle;
//    cublasCreate(&handle);
//	cudaStream_t pStream[numberOfMultiplications];
//	srand((unsigned)time(0));
//	float *h_A, **h_B, **h_C;
//	h_B = (float **) malloc(numberOfMultiplications*sizeof(float *));
//	h_C = (float **) malloc(numberOfMultiplications*sizeof(float *));
//	h_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
//	for(int i = 0 ; i < numberOfMultiplications; i++) {
//		h_B[i] = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
//		h_C[i] = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));
//	}
// //   float *h_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
//	//float *h_D = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
//	//float *h_E = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
//	//float *h_F = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
//	//float *h_G = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
//	//float *h_H = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
//	//float *h_I = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
// //   float *h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));
//	
//
//	float *d_A, **d_B, **d_C;
//	d_B = (float **) malloc(numberOfMultiplications*sizeof(float *));
//	d_C = (float **) malloc(numberOfMultiplications*sizeof(float *));
//
//    cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(float));
//
//	for(int i = 0 ; i < numberOfMultiplications; i++) {
//		cudaMalloc(&d_B[i],nr_rows_B * nr_cols_B * sizeof(float));
//		cudaMalloc(&d_C[i],nr_rows_C * nr_cols_C * sizeof(float));
//	}
//	float X = 345.43;
//	for(int i = 0; i < numberOfMultiplications; i++)
//		cudaStreamCreate(&pStream[i]);
//
//	for (int i = 0 ; i < numberOfMultiplications; i++)
//		for(int j = 0; j < (nr_rows_B * nr_cols_B); j++) 
//			h_B[i][j] = (float)rand()/((float)RAND_MAX/X);
//		//h_A[i] = (float)i;
//
//	//for(int i = 0; i < (nr_rows_B * nr_cols_B); i++) { 
//	//	h_B[i] = (float)rand()/((float)RAND_MAX/X);
//	//	h_D[i] = (float)rand()/((float)RAND_MAX/X);
//	//	h_E[i] = (float)rand()/((float)RAND_MAX/X);
//	//	h_F[i] = (float)rand()/((float)RAND_MAX/X);
//	//	h_G[i] = (float)rand()/((float)RAND_MAX/X);
//	//	h_H[i] = (float)rand()/((float)RAND_MAX/X);
//	//	h_I[i] = (float)rand()/((float)RAND_MAX/X);
//
//	//}
//		//h_B[i] = (float)i;
//
//	for (int i = 0 ; i < numberOfMultiplications; i++)
//		for(int j = 0; j < (nr_rows_C * nr_cols_C); j++) 
//			h_C[i][j] = 0.0;
//
//
//	
//	cudaMemcpy(d_A,h_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyHostToDevice);
//    
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//	cudaEventRecord(start, NULL);
//	
//	for(int i = 0; i < numberOfMultiplications; i++) {
//		cudaMemcpy(d_B[i],h_B[i],nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice);
//	}
//
//	//cudaMemcpy(d_B[0],h_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice);
//	//cudaMemcpy(d_B[1],h_D,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice);
//	//cudaMemcpy(d_B[2],h_E,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice);
//	//cudaMemcpy(d_B[3],h_F,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice);
//	//cudaMemcpy(d_B[4],h_G,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice);
//	//cudaMemcpy(d_B[5],h_H,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice);
//	//cudaMemcpy(d_B[6],h_I,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice);
//
//	for (int i = 0; i < numberOfMultiplications; i++) {
//		cublasSetStream(handle, pStream[i]);
//		//cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, size,1, size, &alpha, (const float**)devAList, lda, (const float**)devBList, ldb, &beta, devCList, ldc, num);
//		cublasSgemv(handle, CUBLAS_OP_N, nr_rows_A, nr_rows_A, alpha, d_A, nr_rows_A, d_B[i], 1, beta, d_C[i], 1);
//	}
//	
//	for(int i= 0 ; i < numberOfMultiplications; i++) 
//		cudaMemcpy(h_C,d_C[i],nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
//	//cudaMemcpy(h_C,d_C[1],nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
//	//cudaMemcpy(h_C,d_C[2],nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
//	//cudaMemcpy(h_C,d_C[3],nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
//	//cudaMemcpy(h_C,d_C[4],nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
//	//cudaMemcpy(h_C,d_C[5],nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
//	//cudaMemcpy(h_C,d_C[6],nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
//	cudaEventRecord(stop, NULL);
//	cudaEventSynchronize(stop);
//	float msecTotal = 0.0f;
//	
//    cudaEventElapsedTime(&msecTotal, start, stop);
//	std::cout << "total time taken "<<msecTotal/(numberOfIterations)<<"\n";
//	
//	cublasDestroy(handle);
// //   std::cout << "A =" << std::endl;
//
//	//for(int i = 0; i < nr_rows_A; ++i){
// //       for(int j = 0; j < nr_rows_B; ++j){
// //           std::cout << h_A[j * nr_rows_A + i] << " ";
// //       }
// //       std::cout << std::endl;
// //   }
//	//std::cout << "B =" << std::endl;
//
//	//for(int i = 0; i < nr_rows_B; ++i){
// //       for(int j = 0; j < nr_cols_B; ++j){
// //           std::cout << h_B[j * nr_rows_B + i] << " ";
// //       }
// //       std::cout << std::endl;
// //   }
// //   std::cout << "C =" << std::endl;
//
//	//for(int i = 0; i < nr_rows_A; ++i){
// //       for(int j = 0; j < 1; ++j){
// //           std::cout << h_C[j * nr_rows_A + i] << " ";
// //       }
// //       std::cout << std::endl;
// //   }
// //   std::cout << std::endl;
//	//Free GPU memory
//    cudaFree(d_A);
//    cudaFree(d_B);
//    cudaFree(d_C);  
//
//    // Free CPU memory
//    free(h_A);
//    free(h_B);
//    free(h_C); 
//	return(0);
//}


/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This example demonstrates how to get better performance by
 * batching CUBLAS calls with the use of using streams
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This example demonstrates how to get better performance by
 * batching CUBLAS calls with the use of using streams
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#if defined(_WIN32)
#include <float.h>
#endif

/* Using updated (v2) interfaces to cublas and cusparse */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "mkl.h"
// Utilities and system includes
#include <helper_cuda.h>
#include "batchCUBLAS.h"
#include <ctime>
const char *sSDKname = "batchCUBLAS";
const int LOOP_COUNT = 100;
//============================================================================================
// Device information utilities
//============================================================================================
cuComplex *A = NULL;
cuComplex *B = NULL;
cuComplex *C = NULL;
//MKL_Complex8 *Am, *Bm, *Cm;
#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

    int getDeviceVersion(void)
    {
        int device;
        struct cudaDeviceProp properties;

        if (cudaGetDevice(&device) != cudaSuccess)
        {
            printf("failed to get device\n");
            return 0;
        }

        if (cudaGetDeviceProperties(&properties, device) != cudaSuccess)
        {
            printf("failed to get properties\n");
            return 0;
        }

        return properties.major * 100 + properties.minor * 10;
    }

    size_t getDeviceMemory(void)
    {
        struct cudaDeviceProp properties;
        int device;

        if (cudaGetDevice(&device) != cudaSuccess)
        {
            return 0;
        }

        if (cudaGetDeviceProperties(&properties, device) != cudaSuccess)
        {
            return 0;
        }

        return properties.totalGlobalMem;
    }
#if defined(__cplusplus)
}
#endif /* __cplusplus */

//============================================================================================
// random utilities
//============================================================================================

template < typename T_ELEM>
void  fillupMatrix(cuComplex *A , int lda , int rows, int cols, int seed = 0);

void fillupMatrix(cuComplex *A , int lda , int rows, int cols, int seed)
{
    for (int j = 0; j < rows; j++)
    {
        for (int i = 0; i < cols; i++)
        {
			A[i + lda*j].x = (float)rand()/((float)RAND_MAX);
			A[i + lda*j].y = (float)rand()/((float)RAND_MAX);



            //A[i + lda*j ] = cuGet<T_ELEM> (((double)(((lda*i+j+seed) % 253)+1))/256.0, ((double)((((cols*i+j) + 123 + seed) % 253)+1))/256.0);
			//A[i + lda*j ] = (cuComplex) (i + lda*j);
        }
    }
}
/* Explicit instantiation */
template void  fillupMatrix<float>(float *A , int lda , int rows, int cols, int seed);
template void  fillupMatrix<double>(double *A , int lda , int rows, int cols, int seed);

/* For debugging */
void printCuType(const char *str, float A)
{
    fprintf(stdout, "%s (0x%08x, %g)", str, floatAsUInt(A), A);
}

void printCuType(const char *str, double A)
{
    fprintf(stdout, "%s (0x%016llx, %g)", str, doubleAsULL(A), A);
}

//============================================================================================
// defines and structures
//============================================================================================

#define CUBLAS_SGEMM_MAX_ULP_ERR    (.3)
#define CUBLAS_DGEMM_MAX_ULP_ERR    (1.e-3)
#define CUBLAS_SGEMM_MAX_RELATIVE_ERR    (6.e-6)
#define CUBLAS_DGEMM_MAX_RELATIVE_ERR    (0.0)
#define CUBLAS_GEMM_TEST_COUNT     (30)
#define BENCH_MATRIX_M              (4096)
#define BENCH_MATRIX_K              (4096)
#define BENCH_MATRIX_N              (1)

enum testMethod
{
    tmRegular,
    tmStream,
    tmBatched
};

struct gemmOpts
{
    int m;
    int n;
    int k;
    testMethod test_method;
    char *elem_type;
    int N;    // number of multiplications
};

template<typename T_ELEM>
struct gemmTestParams
{
    cublasOperation_t transa;
    cublasOperation_t transb;
    int   m;
    int   n;
    int   k;
    cuComplex alpha;
    cuComplex beta;
};

//============================================================================================
// template wrappers for cuda functions
//============================================================================================

static inline cublasStatus_t cublasXgemm(cublasHandle_t handle,
                                         cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                                         float *alpha, const float *A, int lda,
                                         float *B, int ldb, float *beta,
                                         float *C, int ldc)
{
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

static inline cublasStatus_t cublasXgemm(cublasHandle_t handle,
                                         cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                                         double *alpha, const double *A, int lda,
                                         double *B, int ldb, double *beta,
                                         double *C, int ldc)
{
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

static inline cublasStatus_t cublasXgemmBatched(cublasHandle_t handle,
                                                cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                                                float *alpha, const float *Aarray[], int lda,
                                                const float *Barray[], int ldb, float *beta,
                                                float *Carray[], int ldc, int batchCount)
{
#if CUDART_VERSION >= 4010
    return cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
#else
    return CUBLAS_STATUS_SUCCESS;
#endif
}

static inline cublasStatus_t cublasXgemmBatched(cublasHandle_t handle,
                                                cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                                                double *alpha, const double *Aarray[], int lda,
                                                const double *Barray[], int ldb, double *beta,
                                                double *Carray[], int ldc,
                                                int batchCount)
{
#if CUDART_VERSION >= 4010
    return cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
#else
    return CUBLAS_STATUS_SUCCESS;
#endif
}

//============================================================================================
// Primary Application code
//============================================================================================

static int processArgs(int argc, char *argv[], struct gemmOpts *opts)
{
    int error = 0;
    int oldError;
    memset(opts, 0, sizeof(*opts));
    static char default_type[] = "d"; //default double
    opts->elem_type = default_type;
    opts->N = 10;

    while (argc)
    {
        oldError = error;

        if (*argv[0] == SWITCH_CHAR)
        {
            switch (*(argv[0]+1))
            {
                case 'm':
                    opts->m = (int)atol(argv[0]+2);
                    break;

                case 'n':
                    opts->n = (int)atol(argv[0]+2);
                    break;

                case 'k':
                    opts->k = (int)atol(argv[0]+2);
                    break;

                case 'N':
                    opts->N = (int)atol(argv[0]+2);
                    break;

                default:
                    break;
            }
        }

        if (error > oldError)
        {
            fprintf(stderr, "Invalid switch '%c%s'\n",SWITCH_CHAR, argv[0]+1);
        }

        argc -= 1;
        argv++;
    }

    return error;
}

template <typename T_ELEM>
static int TESTGEN(gemm)(const struct gemmOpts *opts,
                         int matrixM, int matrixN, int matrixK, int &numTests,
                         struct gemmTestParams<T_ELEM> *params)
{
    static T_ELEM alpha[] = { cuGet<T_ELEM>(0,0), cuGet<T_ELEM>(-1,-1), cuGet<T_ELEM>(1,-2), cuGet<T_ELEM>(2,-1), cuGet<T_ELEM>(0,-3) };
    static T_ELEM beta[]  = { cuGet<T_ELEM>(0,0), cuGet<T_ELEM>(-1,-1), cuGet<T_ELEM>(1,-2), cuGet<T_ELEM>(2,-1), cuGet<T_ELEM>(0,-3)};

#define NBR_ALPHAS (sizeof(alpha) / sizeof(alpha[0]))
#define NBR_BETAS (sizeof(beta) / sizeof(beta[0]))
    static T_ELEM theAlpha;
    static T_ELEM theBeta;
    static int state;
    static int m;
    static int n;
    static int k;

    if (numTests-- <= 0)
    {
        return -1;
    }

    theAlpha = alpha[cuRand()%NBR_ALPHAS];
    theBeta  = beta[cuRand()%NBR_BETAS];
    params->transa = CUBLAS_OP_T;
    params->transb = CUBLAS_OP_N;
    m = matrixM;
    n = matrixN;
    k = matrixK;
    params->m = m;
    params->n = n;
    params->k = k;
	params->alpha.x = 1.0;
	params->alpha.y = 0.0;
	params->beta.x = 0.0;
	params->beta.y = 0.0;

    printf("#### args: ta=%d tb=%d m=%d n=%d k=%d ",
           (unsigned int)params->transa, (unsigned int)params->transb, params->m, params->n,
           params->k);
     printf("\n");

    m = cuRand() % matrixM;
    n = cuRand() % matrixN;
    k = cuRand() % matrixK;

    state = cuRand() % 9;
    return 0;
}

void fillMatrix(cuComplex *A, int rows, int cols) {
	for(int i = 0; i < rows; i++)
		for(int j = 0; j < cols; j++) {
			A[i*rows + j].x = (float)rand()/((float)RAND_MAX);
			A[i*rows + j].y = (float)rand()/((float)RAND_MAX);
			//A[i*rows + j].x = (float)((i*rows + j) % 4096);
			//A[i*rows + j].y = (float)((i*rows + j) % 4096);
		}
}

void fillVector(cuComplex *A, int dim) {
	int c = 0;
	for(int i = 0 ;i < dim; i++) {
		A[i].x = (float)rand()/((float)RAND_MAX);
		A[i].y = (float)rand()/((float)RAND_MAX);
		//A[i].x = (float)(i % 4096);
		//A[i].y = (float)(i % 4096);
	}
}
  
int test_cgemv(struct gemmOpts &opts, float err, double max_relative_error, cublasHandle_t handle)
{
    
    cudaStream_t *streamArray = 0;
    cublasStatus_t status1, status2, status3;
    cuComplex **devPtrA = 0;
    cuComplex *devPtrB = 0;
    cuComplex *devPtrC = 0;
    cuComplex **devPtrA_dev = NULL;
    cuComplex *devPtrB_dev = NULL;
    cuComplex *devPtrC_dev = NULL;
    int matrixM, matrixN, matrixK;
    int rowsA, rowsB, rowsC;
    int colsA, colsB, colsC;
    int matrixSizeA, matrixSizeB, matrixSizeC;
    int errors;
    double start, stop;

    matrixM = BENCH_MATRIX_M;
    matrixN = BENCH_MATRIX_N;
    matrixK = BENCH_MATRIX_K;

    rowsA = imax(1, matrixM);
    colsA = imax(1, matrixK);
    rowsB = imax(1, matrixK);
    colsB = imax(1, matrixN);
    rowsC = imax(1, matrixM);
    colsC = imax(1, matrixN);

    matrixSizeA = rowsA * colsA;
    matrixSizeB = rowsB * colsB;
    matrixSizeC = rowsC * colsC;

    devPtrA =(cuComplex **)malloc(opts.N * sizeof(*devPtrA));
    devPtrB =(cuComplex *)malloc(opts.N * sizeof(devPtrB));
    devPtrC =(cuComplex *)malloc(opts.N * sizeof(devPtrC));
	cudaError_t err2 = cudaMalloc((void **)&devPtrB, matrixSizeB * sizeof(cuComplex) * opts.N);
	cudaError_t err3 = cudaMalloc((void **)&devPtrC, matrixSizeC * sizeof(cuComplex) * opts.N);
	cudaError_t err1;
	if(err2 != cudaSuccess || err3 != cudaSuccess) {
		        fprintf(stderr, "!!!! cuda memory allocation error\n");

	}
    for (int i = 0; i < opts.N ; i++)
    {
       err1 = cudaMalloc((void **)&devPtrA[i], matrixSizeA * sizeof(devPtrA[0][0]));
	if(err1 != cudaSuccess) {
		        fprintf(stderr, "!!!! cuda memory allocation error\n");

	}

	}
    A  = (cuComplex *)malloc(matrixSizeA * sizeof(A[0]));
    B  = (cuComplex *)malloc(matrixSizeB * sizeof(B[0]) * opts.N);
	C  = (cuComplex *)malloc(matrixSizeC * sizeof(C[0]) * opts.N);
	
    if ((!A) || (!B) || (!C))
    {
        fprintf(stderr, "!!!! system memory allocation error\n");
        return CUBLASTEST_FAILED;
    }

    streamArray = (cudaStream_t *)malloc(opts.N * sizeof(cudaStream_t *));

    for (int i = 0; i < opts.N ; i++)
    {
        if (opts.test_method == tmStream)
        {
            cudaError_t cudaErr = cudaStreamCreate(&streamArray[i]);

            if (cudaErr != cudaSuccess)
            {
                fprintf(stderr, "!!!! cannot create stream\n");
                return CUBLASTEST_FAILED;
            }
        }
        else
        {
            streamArray[i] = 0;
        }
    }

    errors = 0;
    int numTests = 1;
       printf("#### args: lda=%d ldb=%d ldc=%d\n", rowsA, rowsB, rowsC);
		double totalTime = 0.0;
 		cuComplex alph, bet;
		alph.x = 1.0;
		alph.y = 0.0;
		bet.x = 0.0;
		bet.y = 0.0;
       for (int i = 0; i < opts.N ; i++)
        {	
				fillMatrix(A, rowsA, colsA);
				
                status1 = cublasSetMatrix(rowsA, colsA, sizeof(A[0]), A, rowsA, devPtrA[i], rowsA);
				if(status1 != cudaSuccess) {
					fprintf(stderr, "!!!! Set matrix error \n");
				}
 
         }

 
		for(int z = 0 ; z < LOOP_COUNT; z++) {
		fillVector(B, rowsB * opts.N);
        memset(C, 0xFF, matrixSizeC * sizeof(C[0]));


        double flopsCoef = 2.0;
		
        
		status2 = cublasSetVector(rowsB * opts.N, sizeof(cuComplex), B, 1, devPtrB, 1);
		status3 = cublasSetVector(rowsC * opts.N, sizeof(cuComplex), C, 1, devPtrC, 1);
		start = dsecnd();
		int count = 0;
            for (int i = 0; i < opts.N ; i++)
            {

				count++;
                cublasSetStream(handle, streamArray[i]);

				status1 = cublasCgemv(handle, CUBLAS_OP_T, rowsA, rowsA,
                                      &alph, devPtrA[i], rowsA,
									  devPtrB + i*rowsA , 1, &bet, devPtrC + i*rowsA, 1);
				
				
				if (status1 != cudaSuccess)
			  {
				     fprintf(stderr, "!!!! There is something that went terribly wrong.. \n");
					return CUBLASTEST_FAILED;
			  }
	
             }
			
         cudaError_t cudaStatus = cudaThreadSynchronize();

        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "!!!! GPU program execution error on cudaThreadSynchronize : cudaError=%d,(%s)\n", cudaStatus,cudaGetErrorString(cudaStatus));
            return CUBLASTEST_FAILED;
        }

        stop = dsecnd();
		cudaMemcpy(C,devPtrC, rowsC * colsC * sizeof(cuComplex) * opts.N,cudaMemcpyDeviceToHost);
		totalTime = totalTime + (stop-start);
			//Free GPU memory
		}


        fprintf(stdout, "^^^^ elapsed = %10.8f sec \n", totalTime/LOOP_COUNT);

    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);  
	    cublasDestroy(handle);
    cudaDeviceReset();

    // Free CPU memory
    free(A);
    free(B);
    free(C);

    return CUBLASTEST_PASSED;
}

void mklMatrixVector() {
//	srand((unsigned)time(0));
//    
//    int m, n, k, i, j;
//	MKL_Complex8 alph ;
//    MKL_Complex8 alpha, beta;
//	alpha.imag = 1.0;
//	alpha.real = 1.0;
//	alph.real = -1.0;
//	alph.imag = 0.0;
//	beta.imag = 0.0;
//	beta.real = 0.0;
//    printf ("\n This example computes real matrix C=alpha*A*B+beta*C using \n"
//
//            " Intel(R) MKL function dgemm, where A, B, and C are matrices and \n"
//
//            " alpha and beta are double precision scalars\n\n");
//
//    m = 4096, k = 4096, n = 1;
//
//    printf (" Initializing data for matrix multiplication C=A*B for matrix \n"
//
//            " A(%ix%i) and matrix B(%ix%i)\n\n", m, k, k, n);
//
//    //alpha = 1.0; beta = 0.0;
//
//    printf (" Allocating memory for matrices aligned on 64-byte boundary for better \n"
//
//               " performance \n\n");
//
//    Am = (MKL_Complex8 *)mkl_malloc( m*k*sizeof( MKL_Complex8 ), 32 );
//
//    Bm = (MKL_Complex8 *)mkl_malloc( k*n*sizeof( MKL_Complex8 ), 32 );
//
//    Cm = (MKL_Complex8 *)mkl_malloc( m*n*sizeof( MKL_Complex8 ), 32 );
//
//    if (Am == NULL || Bm == NULL || Cm == NULL) {
//
//        printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
//
//        mkl_free(Am);
//
//        mkl_free(Bm);
//
//        mkl_free(Cm);
//
//        return;
//
//    }
//
//    float X = 1234.34f;
//
//    printf (" Intializing matrix data \n\n");
//
//    for (i = 0; i < (m*k); i++) {
//
//		Am[i].real = A[i].x;
//		Am[i].imag = A[i].y;
//
//		//Am[i].real = i;
//		//Am[i].imag = 0;
//
//    }
//
//    for (i = 0; i < (k*n); i++) {
//
//		Bm[i].real = B[i].x;
//		Bm[i].imag = B[i].y;
//
//		//Bm[i].real = i;
//		//Bm[i].imag = 0;
//
//    }
//
//    for (i = 0; i < (m*n); i++) {
//
//		Cm[i].real = 0.0;
//		Cm[i].imag = 0.0;
//
//    }
//
//
//    double time_st = dsecnd();
//
//    for (i=0; i<1; ++i)
//
//    {
//        cblas_cgemv(CblasRowMajor, CblasNoTrans, m, k, &alpha, Am, k, Bm, 1, &beta, Cm, 1);
//    }
//
//    double time_end = dsecnd();
//	float diff;
//	printf("Average time: %.7f secs n", (time_end - time_st)/LOOP_COUNT);
//    printf ("\n Computations completed.\n\n");
//
//	//printf("Matrix A \n");
//	//for (i = 0; i < m*k; i++) {
//	//	printf("%f + i %f ", A[i].x, A[i].y);
//	//	printf("\n");
//	//}
//
//	//printf("Matrix Am \n");
//	//for (i = 0; i < m * k; i++) {
//	//	printf("%f + i %f ", Am[i].real, Am[i].imag);
//	//	printf("\n");
//	//}
//
// //	printf("Matrix B \n");
//	//for (i = 0; i < m; i++) {
//	//	printf("%f + i %f ", B[i].x, B[i].y);
//	//	printf("\n");
//	//}
//
//	//printf("Matrix Bm \n");
//	//for (i = 0; i < m; i++) {
//	//	printf("%f + i %f ", Bm[i].real, Bm[i].imag);
//	//	printf("\n");
//	//}
// //
//	//for(int i = 0; i< m; i++) {
//	//	C[i].x = C[i].x - Cm[i].real;
//	//	C[i].y = C[i].y - Cm[i].imag;
//	//}	
//	cblas_caxpy(m, &alph, ( void *)Cm, 1, ( void *)C, 1);	
//	diff = cblas_scnrm2(m, C, 1);
//	//FILE *mkl, *cuda;
//	//mkl=fopen("E:\\testMKL.txt", "w+");
//	//cuda=fopen("E:\\testCUDA.txt", "w+");
//	//printf("Matrix C \n");
//	//for (i = 0; i < m; i++) {
//	//	fprintf(cuda, "%f + i %f \n", C[i].x, C[i].y);
//	//	//printf("%f + i %f ", C[i].x, C[i].y);
//	//	//printf("\n");
//	//}
//
//	//printf("Matrix Cm \n");
//	//for (i = 0; i < m; i++) {
//	//	fprintf(mkl, "%f + i %f \n", Cm[i].real, Cm[i].imag);
//
//	//}
//	//fclose(mkl);
//	//fclose(cuda);
//	
//	float error = diff/(cblas_scnrm2(m, Cm, 1));
//	printf ("\n error in the matrices %f\n\n", error);
// 
//    mkl_free(Am);
//
//    mkl_free(Bm);
//
//    mkl_free(Cm);
//
////    printf (" Example completed. \n\n");

}

int main(int argc, char *argv[])
{
    struct gemmOpts opts;
    int errors, nTimes, nTotalErrors = 0;
    int status = CUBLASTEST_PASSED;
    printf("Starting...\n\n");

	srand((unsigned)time(0));
    cublasHandle_t handle;

    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stdout, "CUBLAS initialization failed!\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }


    opts.N = 14;
	opts.test_method = tmStream;
    status = test_cgemv(opts, (float)CUBLAS_SGEMM_MAX_ULP_ERR, (double)CUBLAS_SGEMM_MAX_RELATIVE_ERR, handle);
	//mklMatrixVector();
    //cublasDestroy(handle);
    //cudaDeviceReset();

    printf("\nTest Summary\n");
    printf("%d error(s)\n", nTotalErrors);
    exit(nTotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}
