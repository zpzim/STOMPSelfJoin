#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <time.h>
#include <cuComplex.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reverse.h>
#include <thrust/transform_scan.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <float.h>
#include <vector>
#include <unordered_map>
#include <math.h>
#include <cuda_fp16.h>
#include "cuda_profiler_api.h"
#include "STOMP.h"

using std::vector;
using std::unordered_map;
using std::make_pair;

static const unsigned int WORK_SIZE = 512;

//This macro checks return value of the CUDA runtime call and exits
//the application if the call failed.
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//This kernel computes a sliding mean with specified window size and a corresponding prefix sum array (A)
template<class DTYPE>
__global__ void sliding_mean(DTYPE* pref_sum,  size_t window, size_t size, DTYPE* means)
{
    const DTYPE coeff = 1.0 / (DTYPE) window;
    size_t a = blockIdx.x * blockDim.x + threadIdx.x;
    size_t b = blockIdx.x * blockDim.x + threadIdx.x + window;

    if(a == 0){
        means[a] = pref_sum[window - 1] * coeff;
    }
    if(a < size - 1){
        means[a + 1] = (pref_sum[b] - pref_sum[a]) * coeff;
    }
}

// This kernel computes the recipricol sliding standard deviaiton with specified window size, the corresponding means of each element, and the prefix squared sum at each element
// We actually compute the multiplicative inverse of the standard deviation, as this saves us from needing to do a division in the main kernel
template<class DTYPE>
__global__ void sliding_std(DTYPE* cumsumsqr, unsigned int window, unsigned int size, DTYPE* means, DTYPE* stds) {
    const DTYPE coeff = 1 / (DTYPE) window;
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.x * blockDim.x + threadIdx.x + window;
    if (a == 0) {
        stds[a] = 1 / sqrt((cumsumsqr[window - 1] * coeff) - (means[a] * means[a]));
    }
    else if (b < size + window) {
        stds[a] = 1 / sqrt(((cumsumsqr[b - 1] - cumsumsqr[a - 1]) * coeff) - (means[a] * means[a]));
    }
}

template<class DTYPE>
__global__ void sliding_norm(DTYPE* cumsumsqr, unsigned int window, unsigned int size, DTYPE* norms) {
    const DTYPE coeff = 1 / (DTYPE) window;
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.x * blockDim.x + threadIdx.x + window;
    if (a == 0) {
        norms[a] = 1 / sqrt(cumsumsqr[window - 1]);
    }
    else if (b < size + window) {
        norms[a] = 1 / sqrt(cumsumsqr[b - 1] - cumsumsqr[a - 1]);
    }
}

template<class DTYPE>
__global__ void sliding_dfdg(const DTYPE *T, const DTYPE *means, DTYPE *df, DTYPE *dg, const int m, const int n) {
    const DTYPE half = 1.0 / (DTYPE) 2.0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n - 1) {
        df[tid] = (T[tid + m] - T[tid]) * half;
        dg[tid] = (T[tid + m] - means[tid + 1]) + (T[tid] - means[tid]);
    }
}

template<class DTYPE>
void fastinvnorm(vector<DTYPE> &norm, const vector<DTYPE> &mean, const vector<DTYPE> &T, int m) {
    
    DTYPE sum = 0;
    for(int i = 0; i < m; ++i){ 
        sum += pow(T[i] - mean[0],2);
    }
    norm[0] =  sum;
    for(int i = 1; i < norm.size(); ++i) {
            norm[i] = norm[i - 1]  + ((T[i-1] - mean[i-1]) + (T[i + m - 1] - mean[i])) * (T[i + m - 1] - T[i - 1]);
    }
    for(int i = 0; i < norm.size(); ++i) {
        norm[i] = 1.0 / sqrt(norm[i]);
    }
}


template<class DTYPE>
void compute_statistics(const vector<DTYPE> &T_h, const DTYPE *T, DTYPE *norms, DTYPE *df, DTYPE *dg, DTYPE *means, size_t n, size_t m, cudaStream_t s)
{
    square<DTYPE> sqr;
    dim3 grid(ceil(n / (double) WORK_SIZE), 1,1);
    dim3 block(WORK_SIZE, 1, 1);
    
    DTYPE *scratch;
    cudaMalloc(&scratch, sizeof(DTYPE) * (n + m - 1));
    gpuErrchk(cudaPeekAtLastError());
    
    thrust::device_ptr<const DTYPE> dev_ptr_T = thrust::device_pointer_cast(T);
    thrust::device_ptr<DTYPE> dev_ptr_scratch = thrust::device_pointer_cast(scratch);

    // Compute prefix sum in scratch
    thrust::inclusive_scan(thrust::cuda::par.on(s), dev_ptr_T, dev_ptr_T + n + m - 1, dev_ptr_scratch, thrust::plus<DTYPE>());
    gpuErrchk(cudaPeekAtLastError());
    // Use prefix sum to compute sliding mean
    sliding_mean<DTYPE><<<grid, block, 0, s>>>(scratch, m, n, means);
    gpuErrchk(cudaPeekAtLastError());

    sliding_dfdg<DTYPE><<<grid, block, 0, s>>>(T, means, df,dg,m,n);
    // Compute prefix sum of squares in scratch
    thrust::transform_inclusive_scan(thrust::cuda::par.on(s), dev_ptr_T, dev_ptr_T + n + m - 1, dev_ptr_scratch, sqr,thrust::plus<DTYPE>());
    gpuErrchk(cudaPeekAtLastError());
    // Use prefix sum of squares to compute the sliding standard deviation
    vector<DTYPE> norm(n, 0);
    vector<DTYPE> mean(n);
    cudaMemcpy(mean.data(), means, n * sizeof(DTYPE), cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());
    fastinvnorm(norm, mean, T_h, m);
    cudaMemcpy(norms, norm.data(), n * sizeof(DTYPE), cudaMemcpyHostToDevice);
    gpuErrchk(cudaPeekAtLastError());
    cudaStreamSynchronize(s);
    gpuErrchk(cudaPeekAtLastError());
    cudaFree(scratch);
    gpuErrchk(cudaPeekAtLastError());
}

template<class DTYPE>
__global__ void elementwise_multiply_inplace(const DTYPE* A, DTYPE *B, const int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < size) {
       B[tid] *= A[tid];
    }
} 

template<>
__global__ void elementwise_multiply_inplace(const cuDoubleComplex* A, cuDoubleComplex* B, const int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < size) {
       B[tid] = cuCmul(A[tid], B[tid]);
    }
}

// A is input unaligned sliding dot products produced by ifft
// out is the computed vector of distances
template<class DTYPE>
__global__ void normalized_aligned_dot_products(const DTYPE* A, const DTYPE divisor,
                                                const unsigned int m, const unsigned int n,
                                                DTYPE* QT)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a < n) {
        QT[a] = A[a + m - 1] / divisor;
    }
}

template<class DTYPE>
__global__ void populate_reverse_pad(const DTYPE *Q, const DTYPE *means, DTYPE *Q_reverse_pad, const int window_size, const int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < window_size) {
        Q_reverse_pad[tid] = Q[window_size - 1 - tid] - means[0];
    }else if(tid < size){ 
        Q_reverse_pad[tid] = 0;
    }
}

template<class DTYPE, class CUFFT_DTYPE>
void sliding_dot_products_and_distance_profile(DTYPE* T, DTYPE* Q, DTYPE *QT, DTYPE *means, const int size, const int window_len, cudaStream_t s)
{        

    const int n = size - window_len + 1;
    const int cufft_data_size = size / 2 + 1;
    dim3 grid(ceil(n / (float) WORK_SIZE), 1, 1);
    dim3 block(WORK_SIZE, 1, 1);

    cufftHandle fft_plan, ifft_plan;    
    DTYPE *Q_reverse_pad;
    CUFFT_DTYPE *Tc, *Qc;
    cufftPlan1d(&fft_plan, size, CUFFT_FORWARD_PLAN, 1);
    cufftPlan1d(&ifft_plan, size, CUFFT_REVERSE_PLAN, 1);
    cufftSetStream(fft_plan, s);
    cufftSetStream(ifft_plan,s);
    cudaMalloc(&Q_reverse_pad, sizeof(DTYPE) * size);
    cudaMalloc(&Tc, sizeof(CUFFT_DTYPE) * cufft_data_size);
    cudaMalloc(&Qc, sizeof(CUFFT_DTYPE) * cufft_data_size);
    // Compute the FFT of the time series
    CUFFT_FORWARD__(fft_plan, T, Tc);
    gpuErrchk(cudaPeekAtLastError());

    // Reverse and zero pad the query
    populate_reverse_pad<DTYPE><<<dim3(ceil(size / (float) WORK_SIZE),1,1), block, 0, s>>>(Q, means, Q_reverse_pad, window_len, size);
    gpuErrchk(cudaPeekAtLastError());
    
    // Compute the FFT of the query
    CUFFT_FORWARD__(fft_plan, Q_reverse_pad, Qc);
    gpuErrchk(cudaPeekAtLastError());
    
    elementwise_multiply_inplace<<<dim3(ceil(cufft_data_size / (float) WORK_SIZE), 1, 1), block, 0, s>>>(Tc, Qc, cufft_data_size);
    gpuErrchk(cudaPeekAtLastError());

    // Compute the ifft
    // Use the space for the query as scratch space as we no longer need it
    CUFFT_REVERSE__(ifft_plan, Qc, Q_reverse_pad);
    gpuErrchk(cudaPeekAtLastError());
    
    normalized_aligned_dot_products<DTYPE><<<grid, block, 0, s>>>(Q_reverse_pad, size, window_len, n, QT);
    gpuErrchk(cudaPeekAtLastError());

    vector<DTYPE> temp(n);
    cudaMemcpy(temp.data(), QT, n * sizeof(DTYPE), cudaMemcpyDeviceToHost);
    cudaFree(Q_reverse_pad);
    cudaFree(Tc);
    cudaFree(Qc);
    cufftDestroy(fft_plan);
    cufftDestroy(ifft_plan);
} 





// Atomically updates the MP/idxs using a single 64-bit integer. If we do not do this we are unable
// to atomically update both the matrix profile and the indexes without using a critical section and dedicated locks.
__device__ inline void MPatomicMax(volatile unsigned long long int* __restrict__ address, float val, unsigned int idx)
{
    mp_entry loc, loctest;
    loc.floats[0] = val;
    loc.ints[1] = idx;
    loctest.ulong = *address;
    while (loctest.floats[0] < val){
        loctest.ulong = atomicCAS((unsigned long long int*) address, loctest.ulong,  loc.ulong);
    }
}

template<class DTYPE, unsigned int BLOCKSZ, unsigned int tile_height, unsigned int tile_width>
__device__  void initialize_tile_memory(const unsigned long long int* __restrict__ profile, const double* __restrict__ df,
                                              const double* __restrict__ dg, const double* __restrict__ norms,
                                              mp_entry* __restrict__ local_mp_col, mp_entry* __restrict__ local_mp_row,
                                              DTYPE* __restrict__ df_col, DTYPE* __restrict__ df_row, DTYPE* __restrict__ dg_col,
                                              DTYPE* __restrict__ dg_row, float* __restrict__ norm_col, float* __restrict__ norm_row,
                                              const unsigned int n, const unsigned int col_start, const unsigned int row_start)
{
    int global_position = col_start + threadIdx.x;
    int local_position = threadIdx.x;
    while(local_position < tile_width && global_position < n) {
        dg_col[local_position] = dg[global_position];
        df_col[local_position] = df[global_position];
        norm_col[local_position] = norms[global_position];
        local_mp_col[local_position].ulong = profile[global_position];
        local_position += BLOCKSZ;
        global_position += BLOCKSZ;
    }

    global_position = row_start + threadIdx.x;
    local_position = threadIdx.x;
    while(local_position < tile_height && global_position < n) {
        dg_row[local_position] = dg[global_position];
        df_row[local_position] = df[global_position];
        norm_row[local_position] = norms[global_position];
        local_mp_row[local_position].ulong = profile[global_position];
        local_position += BLOCKSZ;
        global_position += BLOCKSZ;
    
    }

}


__device__ inline void MPMax(const float d1, const float d2, const unsigned int i1,
                             const unsigned int i2, float &outd, unsigned int &outi)
{
    if(d1 >= d2) {
        outd = d1;
        outi = i1;
    } else {
        outd = d2;
        outi = i2;
    }

}

// Computes max(a,b) with index and stores the result in a
__device__ inline void MPMax2(float &d1, const float &d2, unsigned int &i1,
                             const unsigned int &i2)
{
    if(d2 > d1) {
        d1 = d2;
        i1 = i2;
    } 
}


// Computes the max of 4 values in a float 4
__device__ inline float max4(const float4 &d, const unsigned int init, unsigned int &idx) {
    float ret = d.x;
    idx = init;
    if(d.y > ret) {
        ret = d.y;
        idx = init + 1;
    }
    if(d.z > ret) {
        ret = d.z;
        idx = init + 2;
    }
    if(d.w > ret) {
        ret = d.w;
        idx = init + 3;
    }
    return ret;
}

// Processes an iteration of the inner loop. Each thread computes 4 distances per iteration (x,y), (x+1,y), (x+1,y+1), and (x+2,y+1)
// This function assumes that the edge cases that occur on the edge of the distance matrix are not present. This is the faster path,
// with less conditional branching.
__device__ inline void do_iteration_unroll_2(int i, int j, int x, int y, int n, float &cov, float &cov2,
                                             float *df_col, float *df_row, float *dg_col, float *dg_row,
                                             float *inorm_col, float *inorm_row, mp_entry *local_mp_col,
                                              mp_entry *local_mp_row) 
{
    float dist_1,dist_2,dist_3;
    unsigned int idx_1,idx_2,idx_3;
    int r = i >> 1;
    int c = j >> 1;
    // Preload the shared memory values we will use into registers
    // We load 2 values per instruction into a float2 vector type
    float2 dfc = reinterpret_cast<float2*>(df_col)[c];
    float2 dgc = reinterpret_cast<float2*>(dg_col)[c];
    float2 inormc = (reinterpret_cast<float2*>(inorm_col)[c]);
    float2 dgr = reinterpret_cast<float2*>(dg_row)[r];
    float2 dfr = reinterpret_cast<float2*>(df_row)[r];
    float2 inormr = reinterpret_cast<float2*>(inorm_row)[r];
    float dfcz = df_col[j + 2];
    float dgcz = dg_col[j + 2];
    float inormcz = inorm_col[j + 2];
    float distx, disty, distz, distw;
    
    // Compute the next set of distances (row y)
    distx = static_cast<float>(cov) * inormc.x * inormr.x;
    disty = static_cast<float>(cov2) * inormc.y * inormr.x;

    // Update the matrix profile, see comment below for explanation
    MPatomicMax((unsigned long long*) (local_mp_col + j), distx, y);
    MPMax(distx, disty, x, x + 1, dist_1, idx_1);
    MPatomicMax((unsigned long long*) (local_mp_row + i), dist_1, idx_1);	

    // Update cov and compute the next distance values (row y + 1)
    cov = cov + dfc.x * dgr.x + dgc.x * dfr.x;
    cov2 = cov2 + dfc.y * dgr.x + dgc.y * dfr.x;

    distz = static_cast<float>(cov) * inormc.y * inormr.y;
    distw = static_cast<float>(cov2) * inormcz * inormr.y; 

    // There are 2 pairs of distances that share the same row and one 
    // pair of distances which share the same column
    // Including the previous updates, we need to perform 3 columnar
    // updates and 2 row updates to the cached matrix profile per iteration
    MPMax(disty, distz, y, y+1, dist_3, idx_3);
    MPMax(distz,distw,x+1, x+2, dist_2, idx_2); 
    MPatomicMax((unsigned long long*) (local_mp_row + i + 1), dist_2, idx_2);
    MPatomicMax((unsigned long long*) (local_mp_col + j + 1), dist_3, idx_3);
    MPatomicMax((unsigned long long*) (local_mp_col + j + 2), distw, y + 1);
    
    // Update cov values for the next iteration
    cov = cov + dfc.y * dgr.y + dgc.y * dfr.y;
    cov2 = cov2 + dfcz * dgr.y + dgcz * dfr.y;
}


// This does one row of work for 4 diagonals in a single thread
__device__ inline void do_unrolled_row4(float &cov1, float &cov2, float &cov3, float &cov4,
                                         float &distcol1, float &distcol2, float &distcol3,
                                         float &distcol4, unsigned int &idxcol1,
                                         unsigned int &idxcol2, unsigned int &idxcol3, unsigned int &idxcol4,
                                         const float &inormcx, const float &inormcy, const float &inormcz,
                                         const float &inormcw, const float &inormr,
                                         const float &df_colx, const float &df_coly, const float &df_colz,
                                         const float &df_colw, const float &dg_colx, const float &dg_coly,
                                         const float &dg_colz, const float &dg_colw, const float &df_row,
                                         const float &dg_row, const int &row, const int &col,
                                         const int &global_row, const int &global_col,
                                         mp_entry* __restrict__ mp_row) {

    float4 dist;

    // Compute the row's distances
    dist.x = static_cast<float>(cov1) * inormcx * inormr;
    dist.y = static_cast<float>(cov2) * inormcy * inormr;
    dist.z = static_cast<float>(cov3) * inormcz * inormr;
    dist.w = static_cast<float>(cov4) * inormcw * inormr;

    // Compute the next covariance values
    cov1 = cov1 + df_colx * dg_row + dg_colx * df_row;
    cov2 = cov2 + df_coly * dg_row + dg_coly * df_row;
    cov3 = cov3 + df_colz * dg_row + dg_colz * df_row;
    cov4 = cov4 + df_colw * dg_row + dg_colw * df_row;

    // Update the column best-so-far values
    MPMax2(distcol1, dist.x, idxcol1, global_row);
    MPMax2(distcol2, dist.y, idxcol2, global_row);
    MPMax2(distcol3, dist.z, idxcol3, global_row);
    MPMax2(distcol4, dist.w, idxcol4, global_row);
    unsigned int idx;

    // We take the maximum of the columns we computed for the row
    // And use that value to check the matrix profile
    float d = max4(dist, global_col, idx);
    MPatomicMax((unsigned long long*) (mp_row + row), d, idx);
}

// Processes 4 iterations of the inner loop. Each thread computes 4 distances per iteration (x,y), (x+1,y), (x+2,y), and (x+3,y)
// This function assumes that the edge cases that occur on the edge of the distance matrix are not present. This is the faster path,
// with less conditional branching.
__device__ inline void do_iteration_unroll_4(int i, int j, int x, int y, int n, float &cov1, float &cov2, float &cov3, float &cov4,
                                             float* __restrict__ df_col, float* __restrict__ df_row, float* __restrict__ dg_col,
                                             float* __restrict__ dg_row, float* __restrict__ inorm_col, float* __restrict__ inorm_row,
                                             mp_entry* __restrict__ local_mp_col, mp_entry* __restrict__ local_mp_row) 
{
    float4 distc = make_float4(CC_MIN, CC_MIN, CC_MIN, CC_MIN);
    float4 distc2 = make_float4(CC_MIN, CC_MIN, CC_MIN, CC_MIN);
    uint4 idxc,idxc2;
    
    // Load row values 2 at a time, load column values 4 at a time
    int r = i >> 1;
    int c = j >> 2;

    // Preload the shared memory values we will use into registers
    // We load 4 values per instruction into a float4 vector type
    float4 dfc = reinterpret_cast<float4*>(df_col)[c];
    float4 dgc = reinterpret_cast<float4*>(dg_col)[c];
    float4 inormc = (reinterpret_cast<float4*>(inorm_col)[c]);
    float4 dfc2 = reinterpret_cast<float4*>(df_col)[c+1];
    float4 dgc2 = reinterpret_cast<float4*>(dg_col)[c+1];
    float4 inormc2 = reinterpret_cast<float4*>(inorm_col)[c+1];

    // Due to a lack of registers on volta, we only load these row values 2 at a time
    float2 dgr = reinterpret_cast<float2*>(dg_row)[r];
    float2 dfr = reinterpret_cast<float2*>(df_row)[r];
    float2 inormr = reinterpret_cast<float2*>(inorm_row)[r];

    // Do rows one at a time:
    // We are computing a tile that looks like this:
    // C:1 2 3 4 5 6 7
    //R1 X X X X
    //R2   X X X X
    //R3     X X X X
    //R4       X X X X
    // For 4 diagonals unrolled 4 times we compute a total of 16 distances.
    // These distances cover 4 possible rows and 7 possible columns, so we need to check the matrix profile
    // 11 times total, once for each row and once for each column
    
    do_unrolled_row4(cov1, cov2, cov3, cov4, distc.x, distc.y, distc.z, distc.w,
                     idxc.x, idxc.y, idxc.z, idxc.w, inormc.x, inormc.y, inormc.z, 
                     inormc.w, inormr.x, dfc.x, dfc.y, dfc.z, dfc.w, dgc.x, dgc.y,
                     dgc.z, dgc.w, dfr.x, dgr.x, i, j, y, x, local_mp_row);

    // Each row's computation allows us to complete a column, the first row completes column 1
    MPatomicMax((unsigned long long*) (local_mp_col + j), distc.x, idxc.x);

    do_unrolled_row4(cov1, cov2, cov3, cov4, distc.y, distc.z, distc.w, distc2.x,
                     idxc.y, idxc.z, idxc.w, idxc2.x, inormc.y, inormc.z, inormc.w,
                     inormc2.x, inormr.y, dfc.y, dfc.z, dfc.w, dfc2.x, dgc.y, dgc.z,
                     dgc.w, dgc2.x, dfr.y, dgr.y, i + 1, j + 1, y + 1, x + 1,
                     local_mp_row);

    // The second row completes column 2
    MPatomicMax((unsigned long long*) (local_mp_col + j + 1), distc.y, idxc.y);

    // Load the values for the next 2 rows
    dgr = reinterpret_cast<float2*>(dg_row)[r + 1];
    dfr = reinterpret_cast<float2*>(df_row)[r + 1];
    inormr = reinterpret_cast<float2*>(inorm_row)[r + 1];

    do_unrolled_row4(cov1, cov2, cov3, cov4, distc.z, distc.w, distc2.x, distc2.y,
                     idxc.z, idxc.w, idxc2.x, idxc2.y, inormc.z, inormc.w, inormc2.x,
                     inormc2.y, inormr.x, dfc.z, dfc.w, dfc2.x, dfc2.y, dgc.z, dgc.w,
                     dgc2.x, dgc2.y, dfr.x, dgr.x, i + 2, j + 2, y + 2, x + 2,
                     local_mp_row);

    // The third row completes column 3
    MPatomicMax((unsigned long long*) (local_mp_col + j + 2), distc.z, idxc.z);

    do_unrolled_row4(cov1, cov2, cov3, cov4, distc.w, distc2.x, distc2.y, distc2.z,
                     idxc.w, idxc2.x, idxc2.y, idxc2.z, inormc.w, inormc2.x, inormc2.y,
                     inormc2.z, inormr.y, dfc.w, dfc2.x, dfc2.y, dfc2.z, dgc.w, dgc2.x,
                     dgc2.y, dgc2.z, dfr.y, dgr.y, i + 3, j + 3, y + 3, x + 3,
                     local_mp_row);
   
    // After the 4th row, we have completed columns 4, 5, 6, and 7
    MPatomicMax((unsigned long long*) (local_mp_col + j + 3), distc.w, idxc.w);
    MPatomicMax((unsigned long long*) (local_mp_col + j + 4), distc2.x, idxc2.x);
    MPatomicMax((unsigned long long*) (local_mp_col + j + 5), distc2.y, idxc2.y);
    MPatomicMax((unsigned long long*) (local_mp_col + j + 6), distc2.z, idxc2.z);
    
}


// Does a single iteration of the inner loop on 2 diagonals, not unrolled
// Checks for the boundary case where only 1 diagonal can be updated
__device__ inline void do_iteration_2diag(int i, int j, int x, int y, int n, float &cov, float &cov2,
                                             float *df_col, float *df_row, float *dg_col, float *dg_row,
                                             float *inorm_col, float *inorm_row, mp_entry *local_mp_col,
                                              mp_entry *local_mp_row) 
{
    float dist_1;
    unsigned int idx_1;
    // Compute the next set of distances (row y)
    float distx = static_cast<float>(cov) * inorm_col[j] * inorm_row[i];
    float disty = static_cast<float>(cov2) * inorm_col[j + 1] * inorm_row[i];
    // Update cov and compute the next distance values (row y)
    cov = cov + df_col[j] * dg_row[i] + dg_col[j] * df_row[i];
    cov2 = cov2 + df_col[j+1] * dg_row[i] + dg_col[j+1] * df_row[i];

    // Update the matrix profile, see comment below for explanation
    MPatomicMax((unsigned long long*) (local_mp_col + j), distx, y);
    if(x + 1 < n) {
        MPatomicMax((unsigned long long*) (local_mp_col + j + 1), disty, y);
        MPMax(distx, disty, x, x + 1, dist_1, idx_1);
        MPatomicMax((unsigned long long*) (local_mp_row + i), dist_1, idx_1);	
    }else {
        MPatomicMax((unsigned long long*) (local_mp_row + i), distx, x);	
    }
}

// Does a single iteration of the inner loop on 4 diagonals per thread, not unrolled
// Checks for the boundary case where only 1, 2, or 3 diagonals can be updated
__device__ inline void do_iteration_4diag(int i, int j, int x, int y, int n, float &cov1, float &cov2,
                                          float &cov3, float &cov4, float *df_col, float *df_row,
                                          float *dg_col, float *dg_row, float *inorm_col, float *inorm_row,
                                          mp_entry *local_mp_col, mp_entry *local_mp_row) 
{
    float dist_1;
    unsigned int idx_1;
    float4 dist;
    // Compute the next set of distances (row y)
    dist.x = static_cast<float>(cov1) * inorm_col[j] * inorm_row[i];
    dist.y = static_cast<float>(cov2) * inorm_col[j + 1] * inorm_row[i];
    dist.z = static_cast<float>(cov3) * inorm_col[j + 2] * inorm_row[i];
    dist.w = static_cast<float>(cov4) * inorm_col[j + 3] * inorm_row[i];

    // Update cov and compute the next distance values (row y)
    cov1 = cov1 + df_col[j] * dg_row[i] + dg_col[j] * df_row[i];
    cov2 = cov2 + df_col[j+1] * dg_row[i] + dg_col[j+1] * df_row[i];
    cov3 = cov3 + df_col[j+2] * dg_row[i] + dg_col[j + 2] * df_row[i];
    cov4 = cov4 + df_col[j+3] * dg_row[i] + dg_col[j + 3] * df_row[i];

    // Update the matrix profile, see comment below for explanation
    MPatomicMax((unsigned long long*) (local_mp_col + j), dist.x, y);
    dist_1 = dist.x;
    idx_1 = x;
    if(x + 1 < n) {
        MPMax(dist_1, dist.y, idx_1, x + 1, dist_1, idx_1);
        MPatomicMax((unsigned long long*) (local_mp_col + j + 1), dist.y, y);
    }
    if(x + 2 < n) {
        MPMax(dist_1, dist.z, idx_1, x + 2, dist_1, idx_1);
        MPatomicMax((unsigned long long*) (local_mp_col + j + 2), dist.z, y);
    }
    if(x + 3 < n) {
        MPMax(dist_1, dist.w, idx_1, x + 3, dist_1, idx_1);
        MPatomicMax((unsigned long long*) (local_mp_col + j + 3), dist.w, y);
    }
    MPatomicMax((unsigned long long*) (local_mp_row + i), dist_1, idx_1);	
}

//Computes the matrix profile given the sliding dot products for the first query and the precomputed data statisics
template<class DTYPE, unsigned int BLOCKSZ>
__global__ void __launch_bounds__(BLOCKSZ, 512 * 2  / BLOCKSZ)
WavefrontUpdateSelfJoin(const double* __restrict__ Cov, const double* __restrict__ df,
                        const double* __restrict__ dg, const double* __restrict__ norms,
                        unsigned long long* __restrict__ profile, const unsigned int m,
                        const unsigned int n, int startPos, int numDevices)
{
    // tile_height must be a multiple of 4
    // Tuned for V100
    const int tile_height = 200;
    const int tile_width = tile_height + BLOCKSZ * 4;
    __shared__ mp_entry local_mp_col[tile_width];
    __shared__ mp_entry local_mp_row[tile_height];
    __shared__ float df_col[tile_width];
    __shared__ float dg_col[tile_width];
    __shared__ float inorm_col[tile_width];
    __shared__ float df_row[tile_height];
    __shared__ float dg_row[tile_height];
    __shared__ float inorm_row[tile_height];

    // This is the index of the meta-diagonal that this thread block will work on
    int meta_diagonal_idx = blockIdx.x * numDevices + startPos;

    // The first threads are acutally computing the trivial match between the same subsequence
    // we exclude these from the calculation
    const int exclusion = (m / 4);
    int tile_start_x = meta_diagonal_idx * (BLOCKSZ * 4) + exclusion;
    int tile_start_y = 0;
    
    // x is the global column of the distance matrix
    // y is the global row of the distance matrix
    // localX, localY are the local coordinates of the thread position in the tile it is working on
    int x = tile_start_x + threadIdx.x * 4;
    int y = 0;

    // Each thread updates 2 diagonals at once
    float cov1, cov2, cov3, cov4;
    
    // Load the first dot product values
    if (x < n) {
        cov1 = Cov[x];
    }
    
    if (x + 1 < n) {
        cov2 = Cov[x + 1];
    }
    
    if (x + 2 < n) {
        cov3 = Cov[x + 2];
    }

    if(x + 3 < n) {
        cov4 = Cov[x + 3]; 
    }

    /////////////////////////////////////    
    // Main loop
    /////////////////////////////////////
    // Each threadblock finds all the distances on a 'metadiagonal'
    // We use a tiled approach for each thread block
    // The tiles are horizontal slices of the diagonal, think of a parallelogram cut
    // from a diagonal slice of the distance matrix 
    // Each thread starts on the first row and works its way down-right towards right
    // side of the distance matrix
    while (tile_start_x < n)
    {
        // Initialize the next tile's shared memory
        initialize_tile_memory<float, BLOCKSZ, tile_height, tile_width>(profile, df, dg, norms, local_mp_col, local_mp_row,
                                                                        df_col, df_row, dg_col, dg_row, inorm_col, inorm_row,
                                                                        n, tile_start_x, tile_start_y);
        // Start of new tile, sync
        __syncthreads();

        // There are 2 pathways here, most of the time we take the fast path (top),
        // the last block will take the slower path as well as the fast path (bottom)
        if(x + tile_height < n) {
            for(int i = 0, j = threadIdx.x << 2; i < tile_height; i+=4, j+=4) {
                do_iteration_unroll_4(i,j,x + i,y + i,n, cov1,cov2,cov3,cov4,df_col, df_row, dg_col, dg_row, inorm_col, inorm_row, local_mp_col, local_mp_row);
            }
            x += tile_height;
            y += tile_height;
        } else {
            int localX = threadIdx.x << 2;
            int localY = 0;
            while(x < n) {
                do_iteration_4diag(localY,localX,x,y,n, cov1,cov2,cov3,cov4, df_col, df_row, dg_col, dg_row, inorm_col, inorm_row, local_mp_col, local_mp_row);
                ++x;
                ++y;
                ++localX;
                ++localY;
            } 
        }

        // After this sync, the caches will be updated with the best so far values for this tile
        __syncthreads();

        // If we updated any values in the cached MP, try to push them to the global "master" MP
        int global_position = tile_start_x + threadIdx.x;
        int local_position = threadIdx.x;
        while(local_position < tile_width && global_position < n) {
            mp_entry e = local_mp_col[local_position];
        	MPatomicMax(profile + global_position, e.floats[0], e.ints[1]);
            global_position += BLOCKSZ;
            local_position += BLOCKSZ;
        }

        global_position = tile_start_y + threadIdx.x;
        local_position = threadIdx.x;
        while(local_position < tile_height && global_position < n) {
            mp_entry e = local_mp_row[local_position];
            MPatomicMax(profile + global_position, e.floats[0], e.ints[1]);
            global_position += BLOCKSZ;
            local_position += BLOCKSZ;
        }
        
        // Update the tile position
        tile_start_x += tile_height;
        tile_start_y += tile_height;

        // Make sure our updates were committed before we pull in the next tile
        __threadfence_block();
    }
    

}

__global__ void cross_correlation_to_ed(float *profile, unsigned int n, unsigned int m) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n) {
        profile[tid] = sqrt(max(2*(1 - profile[tid]), 0.0)) * sqrt((double)m);
    }
}

// The STOMP algorithm
template<class DTYPE, class CUFFT_DTYPE>
void do_STOMP(const vector<DTYPE> &T_h, vector<float> &profile_h, vector<unsigned int> &profile_idx_h, const unsigned int m, const vector<int> &devices) {
    if(devices.empty()) {
        printf("Error: no gpu provided\n");
        exit(0);
    }
    
    size_t n = T_h.size() - m + 1;
    
    unordered_map<int, DTYPE*> T_dev, QT_dev, means, norms, df, dg;
    unordered_map<int, float*> profile_dev;
    unordered_map<int, unsigned long long int*> profile_merged;
    unordered_map<int, unsigned int*> profile_idx_dev;
    unordered_map<int, cudaEvent_t> clocks_start, clocks_end;
    unordered_map<int, cudaStream_t> streams;

    // Allocate and initialize memory
    for (auto device : devices) {
        cudaSetDevice(device);
        gpuErrchk(cudaPeekAtLastError());
        T_dev.insert(make_pair(device, (DTYPE*) 0));
        QT_dev.insert(make_pair(device, (DTYPE*) 0));
        means.insert(make_pair(device, (DTYPE*) 0));
        norms.insert(make_pair(device, (DTYPE*) 0));
        df.insert(make_pair(device, (DTYPE*) 0));
        dg.insert(make_pair(device, (DTYPE*) 0));
        profile_dev.insert(make_pair(device,(float*) NULL));
        profile_merged.insert(make_pair(device,(unsigned long long int*) NULL));
        profile_idx_dev.insert(make_pair(device,(unsigned int *) NULL));


        cudaMalloc(&T_dev.at(device), sizeof(DTYPE) * T_h.size());
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&profile_dev.at(device), sizeof(float) * profile_h.size());
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&profile_idx_dev.at(device), sizeof(unsigned int) * profile_idx_h.size());
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&QT_dev.at(device), sizeof(DTYPE) * profile_h.size());
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&means.at(device), sizeof(DTYPE) * profile_h.size());
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&norms.at(device), sizeof(DTYPE) * profile_h.size());
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&df.at(device), sizeof(DTYPE) * profile_h.size());
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&dg.at(device), sizeof(DTYPE) * profile_h.size());
        gpuErrchk(cudaPeekAtLastError());
        cudaMalloc(&profile_merged.at(device), sizeof(unsigned long long int) * n);
        gpuErrchk(cudaPeekAtLastError());
        cudaEvent_t st, ed;
        cudaEventCreate(&ed);
        gpuErrchk(cudaPeekAtLastError());
        cudaEventCreate(&st);
        gpuErrchk(cudaPeekAtLastError());
        clocks_start.emplace(device, st);
        clocks_end.emplace(device, ed);
        cudaStream_t s;
        cudaStreamCreate(&s);
        gpuErrchk(cudaPeekAtLastError());
        streams.emplace(device, s);
    }

    MPIDXCombine combiner;
    int num_workers = ceil(ceil((n - (m / 4)) / (float) devices.size()) / 4.0);
    
    // Asynchronously copy relevant data, precompute statistics, generate partial matrix profile
    int count = 0;
    for (auto &device : devices) {
        cudaSetDevice(device);
        cudaMemcpyAsync(T_dev[device], T_h.data(), sizeof(DTYPE) * T_h.size(), cudaMemcpyHostToDevice, streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
        cudaMemcpyAsync(profile_dev[device], profile_h.data(), sizeof(float) * profile_h.size(), cudaMemcpyHostToDevice, streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
        cudaMemcpyAsync(profile_idx_dev[device], profile_idx_h.data(), sizeof(unsigned int) * profile_idx_h.size(), cudaMemcpyHostToDevice, streams.at(device));
        gpuErrchk(cudaPeekAtLastError());

        // Computing the statistics for each device is overkill, but it avoids needing to do some staging on the host if P2P transfer doesn't work
        compute_statistics<DTYPE>(T_h, T_dev[device], norms[device], df[device], dg[device], means[device], n, m, streams.at(device));
        sliding_dot_products_and_distance_profile<DTYPE, CUFFT_DTYPE>(T_dev[device], T_dev[device], QT_dev[device], means[device], T_h.size(), m, streams.at(device));
         
        thrust::device_ptr<unsigned long long int> ptr = thrust::device_pointer_cast(profile_merged[device]);
        thrust::transform(thrust::cuda::par.on(streams.at(device)), profile_dev[device], profile_dev[device] + n, profile_idx_dev[device], profile_merged[device], combiner);
        //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        gpuErrchk(cudaPeekAtLastError());
        printf("Start main kernel on GPU %d\n", device);

        cudaEventRecord(clocks_start[device], streams.at(device));
        WavefrontUpdateSelfJoin<float, WORK_SIZE><<<dim3(ceil(num_workers / (double) WORK_SIZE), 1, 1),dim3(WORK_SIZE, 1,1), 0, streams.at(device)>>>(QT_dev[device], df[device], dg[device], norms[device], profile_merged[device], m, n, count, devices.size());
        cudaEventRecord(clocks_end[device], streams.at(device));
        ++count;
    }
   
    float time;
    for(auto &device : devices) {
        cudaSetDevice(device);
        gpuErrchk(cudaPeekAtLastError());
        cudaStreamSynchronize(streams.at(device));
        cudaEventElapsedTime(&time, clocks_start[device], clocks_end[device]);
        gpuErrchk(cudaPeekAtLastError());
        cudaEventDestroy(clocks_start.at(device));
        cudaEventDestroy(clocks_end.at(device));
        printf("Device %d took %f seconds\n", device, time / 1000);
    }

    printf("Finished STOMP to generate partial matrix profile of size %lu on %lu devices:\n", n, devices.size());

    // Free unneeded resources
    for (auto &device : devices) {
        cudaSetDevice(device);
        gpuErrchk(cudaPeekAtLastError());
        cudaFree(T_dev[device]);
        gpuErrchk(cudaPeekAtLastError());
        // Keep the profile for the first device as a staging area for the final result
        if (device != devices.at(0)) { 
            cudaFree(profile_dev[device]);
            gpuErrchk(cudaPeekAtLastError());
            cudaFree(profile_idx_dev[device]);
            gpuErrchk(cudaPeekAtLastError());
        }
        cudaFree(QT_dev[device]);
        gpuErrchk(cudaPeekAtLastError());
        cudaFree(means[device]);
        gpuErrchk(cudaPeekAtLastError());
        cudaFree(norms[device]);
        gpuErrchk(cudaPeekAtLastError());
        cudaStreamDestroy(streams.at(device));
        gpuErrchk(cudaPeekAtLastError());
    }
   

    // Consolidate the partial matrix profiles to a single vector using the first gpu 
    printf("Merging partial matrix profiles into final result\n");
    vector<unsigned long long int> partial_profile_host(n);
    cudaSetDevice(devices.at(0));
    gpuErrchk(cudaPeekAtLastError());
    auto ptr_profile = thrust::device_ptr<float>(profile_dev[devices.at(0)]);
    auto ptr_index = thrust::device_ptr<unsigned int>(profile_idx_dev[devices.at(0)]);
    auto ptr_merged = thrust::device_ptr<unsigned long long int>(profile_merged[devices.at(0)]);
    auto iter_begin = thrust::make_zip_iterator(thrust::make_tuple(ptr_profile, ptr_index, ptr_merged));
    auto iter_end = thrust::make_zip_iterator(thrust::make_tuple(ptr_profile + n, ptr_index + n, ptr_merged + n));
    for(int i = 0; i < devices.size(); ++i) {
        cudaSetDevice(devices.at(i));
        gpuErrchk(cudaPeekAtLastError());
        if (i != 0) {
            cudaMemcpy(partial_profile_host.data(), profile_merged[devices.at(i)], n * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
            gpuErrchk(cudaPeekAtLastError());
            cudaFree(profile_merged[devices.at(i)]);
            gpuErrchk(cudaPeekAtLastError());
            cudaSetDevice(devices.at(0));
            gpuErrchk(cudaPeekAtLastError());
            cudaMemcpy(profile_merged[0], partial_profile_host.data(), n * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
            gpuErrchk(cudaPeekAtLastError());
        }
        thrust::for_each(iter_begin, iter_end, max_with_index());
        gpuErrchk(cudaPeekAtLastError());
    }
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    cudaSetDevice(devices.at(0));
    gpuErrchk(cudaPeekAtLastError());
         
    // Compute the final distance calculation to convert cross correlation computed earlier into euclidean distance
    cross_correlation_to_ed<<<dim3(ceil(n / (float) WORK_SIZE), 1, 1), dim3(WORK_SIZE, 1, 1)>>>(profile_dev[devices.at(0)], n, m); 
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(profile_idx_h.data(), profile_idx_dev[devices.at(0)], sizeof(unsigned int) * n, cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(profile_h.data(), profile_dev[devices.at(0)], sizeof(float) * n, cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());
    cudaFree(profile_idx_dev[devices.at(0)]);
    gpuErrchk(cudaPeekAtLastError());
    cudaFree(profile_dev[devices.at(0)]);
    gpuErrchk(cudaPeekAtLastError());
    cudaFree(profile_merged[devices.at(0)]);
    gpuErrchk(cudaPeekAtLastError());

}

//Reads input time series from file
template<class DTYPE>
void readFile(const char* filename, vector<DTYPE>& v, const char *format_str) 
{
    FILE* f = fopen( filename, "r");
    if(f == NULL){
        printf("Unable to open %s for reading, please make sure it exists\n", filename);
        exit(0);
    }
    DTYPE num;
    while(!feof(f)){
            fscanf(f, format_str, &num);
            v.push_back(num);
        }
    v.pop_back();
    fclose(f);
}
    


int main(int argc, char** argv) {

    if(argc < 5) {
        printf("Usage: STOMP <window_len> <input file> <profile output file> <index output file> [Optional: list of GPU device numbers to run on]\n");
        exit(0);
    }

    int window_size = atoi(argv[1]);
    
    vector<double> T_h;
    readFile<double>(argv[2], T_h, "%lf");
    int n = T_h.size() - window_size + 1;
    vector<float> profile(n, CC_MIN);
    vector<unsigned int> profile_idx(n, 0);
    
    cudaFree(0);
    
    vector<int> devices;
    
    if(argc == 5) {
        // Use all available devices 
        int num_dev;
        cudaGetDeviceCount(&num_dev);
        for(int i = 0; i < num_dev; ++i){ 
            devices.push_back(i);
        }
    } else {
        // Use the devices specified
        int x = 5;
        while (x < argc) {
            devices.push_back(atoi(argv[x]));
            ++x;
        }
    }
    
    printf("Starting STOMP\n");
     
    do_STOMP<double, cuDoubleComplex>(T_h, profile, profile_idx, window_size, devices);
    
    printf("Now writing result to files\n");
    FILE* f1 = fopen( argv[3], "w");
    FILE* f2 = fopen( argv[4], "w");
    for(int i = 0; i < profile.size(); ++i){
        fprintf(f1, "%f\n", profile[i]);
        fprintf(f2, "%u\n", profile_idx[i] + 1);
    }
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaDeviceReset());
    fclose(f1);
    fclose(f2);
    printf("Done\n");
    return 0;
}



