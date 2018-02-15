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

#define __CUFFT_TYPE__ cuDoubleComplex
#define CUFFT_FORWARD__ cufftExecD2Z
#define CUFFT_REVERSE__ cufftExecZ2D
#define CUFFT_FORWARD_PLAN CUFFT_D2Z
#define CUFFT_REVERSE_PLAN CUFFT_Z2D
#define CC_MIN -FLT_MAX

typedef union  {
  float floats[2];                 // floats[0] = lowest
  unsigned int ints[2];                     // ints[1] = lowIdx
  unsigned long long int ulong;    // for atomic update
} mp_entry;


//For computing the prefix squared sum
template<class DTYPE>
struct square
{
	__host__ __device__
	DTYPE operator()(DTYPE x)
	{
		return x * x;
	}
};

struct MPIDXCombine
{
	__host__ __device__
	unsigned long long int operator()(double x, unsigned int idx){
		mp_entry item;
		item.floats[0] = (float) x;
		item.ints[1] = idx;
		return item.ulong;
	}
};

//Returns the maximum between 2 matrix profile candidates, also records the index of the minumum value
//For constructing the final matrix profile
struct max_with_index
{

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{

		// A[i] = min(A[i], B[i]);
		// C[i] = A[i] == min(A[i], B[i]) ? i : C[i];
		mp_entry other;
		other.ulong = thrust::get<2>(t);
        if(thrust::get<0>(t) < other.floats[0]){
			thrust::get<0>(t) = other.floats[0];
			thrust::get<1>(t) = other.ints[1];
		}

	}
};

#endif

