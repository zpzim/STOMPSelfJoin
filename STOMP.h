#ifndef __STOMP__H_
#define __STOMP__H_

#include <cuComplex.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reverse.h>
#include <thrust/transform_scan.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>

#ifdef __SINGLE_PREC__
#define DATA_TYPE float
#define __CUFFT_TYPE__ cuComplex
#define CUFFT_FORWARD__ cufftExecR2C
#define CUFFT_REVERSE__ cufftExecC2R
#define CUFFT_FORWARD_PLAN CUFFT_R2C
#define CUFFT_REVERSE_PLAN CUFFT_C2R
#define CMUL cuCmulf
#define _MAX_VAL_ FLT_MAX
#else
#define DATA_TYPE double
#define __CUFFT_TYPE__ cuDoubleComplex
#define CUFFT_FORWARD__ cufftExecD2Z
#define CUFFT_REVERSE__ cufftExecZ2D
#define CUFFT_FORWARD_PLAN CUFFT_D2Z
#define CUFFT_REVERSE_PLAN CUFFT_Z2D
#define CMUL cuCmul
#define _MAX_VAL_ DBL_MAX
#endif


//For computing the prefix squared sum
struct square
{
	__host__ __device__
	double operator()(double x)
	{
		return x * x;
	}
};

//For element wise multiplication of complex values
struct multiply
{
	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<0>(t) = CMUL(thrust::get<0>(t), thrust::get<1>(t));
	}


};

//Returns the minimum between 2 matrix profile candidates, also records the index of the minumum value
//For constructing each thread's matrix profile
struct minWithIndex
{
	const unsigned int i;
    minWithIndex(unsigned int _i) : i(_i) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{

		// A[i] = min(A[i], B[i]);
		// C[i] = A[i] == min(A[i], B[i]) ? i : C[i];

		DATA_TYPE x = min(thrust::get<0>(t), thrust::get<1>(t));
		thrust::get<0>(t) = x;
		thrust::get<2>(t) = thrust::get<1>(t) == x ? i : thrust::get<2>(t);

	}
};


//Returns the minimum between 2 matrix profile candidates, also records the index of the minumum value
//For constructing the final matrix profile
struct minWithIndex2
{

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{

		// A[i] = min(A[i], B[i]);
		// C[i] = A[i] == min(A[i], B[i]) ? i : C[i];

		DATA_TYPE x = min(thrust::get<0>(t), thrust::get<1>(t));
		thrust::get<0>(t) = x;
		thrust::get<2>(t) = thrust::get<1>(t) == x ? thrust::get<3>(t) : thrust::get<2>(t);

	}
};

struct maxWithIndex
{
	const unsigned int i;
    maxWithIndex(unsigned int _i) : i(_i) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{

		// A[i] = max(A[i], B[i]);
		// C[i] = A[i] == min(A[i], B[i]) ? i : C[i];

		DATA_TYPE x = max(thrust::get<0>(t), thrust::get<1>(t));
		thrust::get<0>(t) = x;
		thrust::get<2>(t) = thrust::get<1>(t) == x ? i : thrust::get<2>(t);

	}
};


struct maxWithIndex2
{

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{

		// A[i] = min(A[i], B[i]);
		// C[i] = A[i] == min(A[i], B[i]) ? i : C[i];

		DATA_TYPE x = max(thrust::get<0>(t), thrust::get<1>(t));
		thrust::get<0>(t) = x;
		thrust::get<2>(t) = thrust::get<1>(t) == x ? thrust::get<3>(t) : thrust::get<2>(t);

	}
};

void STOMPinit(int size);
void readFile(const char* filename, thrust::host_vector<double>& v);
__global__ void slidingMean(double* A,  int window, unsigned int size, double* Means);
__global__ void slidingStd(double* squares, unsigned int window, unsigned int size, double* Means, double* stds);
__global__ void CalculateDistProfile(double* QT, double* D, double* Means, double* stds, int m, int start, int n);
__global__ void CalculateDistProfileMax(double* QT, double* D, double* Means, double* stds, int m, int start, int n);
__global__ void UpdateQT(double* QT, double* QTtemp, double* QTb, double* Ta, double* Tb, unsigned int i, unsigned int m, unsigned int sz);
void SlidingDotProducts(const thrust::device_vector<double>& Q, const thrust::device_vector<double>& T, thrust::device_vector<double>&  P);
void STOMP(thrust::device_vector<double>& Ta, thrust::device_vector<double>& Tb, unsigned int m,		thrust::device_vector<double>& profile, thrust::device_vector<unsigned int>& profileIdxs, int exclusion, int MaxJoin);

#endif

