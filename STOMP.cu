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

#include "cuda_profiler_api.h"
#include "STOMP.h"

using std::vector;
using std::unordered_map;
using std::make_pair;

static const unsigned int WORK_SIZE = 512;

// These parameters must be tuned for a specific architecture

// By default they are tuned for Volta (V100)
static const unsigned int AMT_UNROLL = 2;
static const unsigned int TILE_HEIGHT_ADJUSTMENT = 4;

//Pascal (P100)
//static const unsigned int AMT_UNROLL = 16;
//static const unsigned int TILE_HEIGHT_ADJUSTMENT = 2;

// Kepler (K80/K40/K20)
// on Kepler, these parameters do not affect the runtime as much because the bottleneck
// is elsewhere
//static const unsigned int AMT_UNROLL = 4;
//static const unsigned int TILE_HEIGHT_ADJUSTMENT = 4;

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
    const DTYPE inv_m = 1.0 / (DTYPE) m;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n - 1) {
        df[tid] = (T[tid + m] - T[tid]) * inv_m;
        dg[tid] = (T[tid + m] - means[tid + 1]) + (T[tid] - means[tid]);
    }
}


template<class DTYPE>
void compute_statistics(const DTYPE *T, DTYPE *norms, DTYPE *df, DTYPE *dg, DTYPE *means, size_t n, size_t m, cudaStream_t s)
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
    sliding_norm<DTYPE><<<grid, block, 0, s>>>(scratch, m, n, norms);
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
        QT[a] = A[a + m - 1];
    }
}

template<class DTYPE>
__global__ void populate_reverse_pad(const DTYPE *Q, const DTYPE *means, DTYPE *Q_reverse_pad, const int window_size, const int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < window_size) {
        Q_reverse_pad[tid] = Q[window_size - 1 - tid] - means[window_size - 1 - tid];
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
    
    cudaFree(Q_reverse_pad);
    cudaFree(Tc);
    cudaFree(Qc);
    cufftDestroy(fft_plan);
    cufftDestroy(ifft_plan);
} 





//Atomically updates the MP/idxs using a single 64-bit integer. We lose a small amount of precision in the output, if we do not do this we are unable
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
                                              DTYPE* __restrict__ dg_row, DTYPE* __restrict__ norm_col, DTYPE* __restrict__ norm_row,
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


__device__ inline void MPMax(const float d1, const float d2, const unsigned int i1, const unsigned int i2, float &outd, unsigned int &outi) {
    if(d1 >= d2) {
        outd = d1;
        outi = i1;
    } else {
        outd = d2;
        outi = i2;
    }

}

// Processes an iteration of the inner loop. Each thread computes 4 distances per iteration (x,y), (x+1,y), (x+1,y+1), and (x+2,y+1)
// This function assumes that the edge cases that occur on the edge of the distance matrix are not present. This is the faster path,
// with less conditional branching.
__device__ inline void do_iteration_unroll_2(int i, int j, int x, int y, int n, double &cov, double &cov2,
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
    float2 inormc = reinterpret_cast<float2*>(inorm_col)[c];
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
    cov = cov - dfc.x * dgr.x + dgc.x * dfr.x;
    cov2 = cov2 - dfc.y * dgr.x + dgc.y * dfr.x;

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
    cov = cov - dfc.y * dgr.y + dgc.y * dfr.y;
    cov2 = cov2 - dfcz * dgr.y + dgcz * dfr.y;
}

// The function above, but now checks for edge cases
__device__ void inline do_iteration_unroll_2_check(int i, int j, int x, int y, int n, double &qt, double &qt2, float *A_low, float *A_high, float *B_low, float *B_high,
			          float *mean_x, float *mean_y, float *inv_std_x, float *inv_std_y, mp_entry *localMPMain, mp_entry *localMPOther) 
{ 
            float meanx2,stdx2,ahigh2,alow2;
            int t = i >> 1;
            int k = j >> 1;
            // We may load some extra uninitialized values here,
	    // But it doesn't matter as we won't use them to update
            // the matrix profile
	    float2 blow = reinterpret_cast<float2*>(B_low)[t];
	    float2 bhigh = reinterpret_cast<float2*>(B_high)[t];
	    float2 meany = reinterpret_cast<float2*>(mean_y)[t];
	    float2 stdy = reinterpret_cast<float2*>(inv_std_y)[t];
            float2 ahigh = reinterpret_cast<float2*>(A_high)[k];
            float2 alow = reinterpret_cast<float2*>(A_low)[k];
	    float2 stdx = reinterpret_cast<float2*>(inv_std_x)[k];
	    float2 meanx = reinterpret_cast<float2*>(mean_x)[k];
            if(x + 2 < n) {
                meanx2 = mean_x[j + 2];
		    stdx2 = inv_std_x[j + 2];
		        ahigh2 = A_high[j + 2];
                alow2 = A_low[j + 2];
	        }
            float distx, disty, distz, distw;
    	    distx = (static_cast<float>(qt) - (meanx.x * meany.x)) * stdx.x * stdy.x;
            disty = (static_cast<float>(qt2) - (meanx.y * meany.x)) * stdx.y * stdy.x;
	    qt = qt - alow.x * blow.x + ahigh.x * bhigh.x;
            qt2 = qt2 - alow.y * blow.x + ahigh.y * bhigh.x;
            distz = (static_cast<float>(qt) - (meanx.y * meany.y)) * stdx.y * stdy.y;
            distw = (static_cast<float>(qt2) - (meanx2 * meany.y)) * stdx2 * stdy.y;
	        qt = qt - alow.y * blow.y + ahigh.y * bhigh.y;
            qt2 = qt2 - alow2 * blow.y + ahigh2 * bhigh.y;
            float dist_1,dist_2,dist_3;
            unsigned int idx_1,idx_2,idx_3;
		    MPatomicMax((unsigned long long*) (localMPMain + j), distx, y);
            
            if(x + 2 < n) {
                MPMax(distx, disty, x, x + 1, dist_1, idx_1);
		MPatomicMax((unsigned long long*) (localMPOther + i), dist_1, idx_1);	
		if(y + 1 < n) {
                	MPMax(disty, distz, y, y+1, dist_3, idx_3);
                	MPMax(distz,distw,x+1, x+2, dist_2, idx_2); 
			MPatomicMax((unsigned long long*) (localMPOther + i + 1), dist_2, idx_2);
			MPatomicMax((unsigned long long*) (localMPMain + j + 1), dist_3, idx_3);
			MPatomicMax((unsigned long long*) (localMPMain + j + 2), distw, y + 1);
		} else {
			MPatomicMax((unsigned long long*) (localMPMain + j + 1), disty, y);
		}
            } else if (x + 1 < n) {
		
                MPMax(distx, disty, x, x + 1, dist_1, idx_1);
		MPatomicMax((unsigned long long*) (localMPOther + i), dist_1, idx_1);	
                if( y + 1 < n) {	
			MPMax(disty, distz, y, y+1, dist_3, idx_3);
		        MPatomicMax((unsigned long long*) (localMPMain + j + 1), dist_3, idx_3);
		        MPatomicMax((unsigned long long*) (localMPOther + i + 1), distz, x + 1);
		} else {
		        MPatomicMax((unsigned long long*) (localMPMain + j + 1), disty, y);
		}
            } else {
		        MPatomicMax((unsigned long long*) (localMPOther + i + 1), distx, x);

            }
}


//Computes the matrix profile given the sliding dot products for the first query and the precomputed data statisics
template<class DTYPE, unsigned int BLOCKSZ, unsigned int UNROLL_COUNT>
__global__ void __launch_bounds__(BLOCKSZ, 2048  / BLOCKSZ)
WavefrontUpdateSelfJoin(const double* __restrict__ Cov, const double* __restrict__ df,
                        const double* __restrict__ dg, const double* __restrict__ norms,
                        unsigned long long* __restrict__ profile, const unsigned int m,
                        const unsigned int n, int startPos, int numDevices)
{
    const int tile_height = BLOCKSZ / TILE_HEIGHT_ADJUSTMENT;
    const int tile_width = tile_height + BLOCKSZ * 2;
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
    int tile_start_x = meta_diagonal_idx * (BLOCKSZ * 2) + exclusion;
    int tile_start_y = 0;
    
    // x is the global column of the distance matrix
    // y is the global row of the distance matrix
    // localX, localY are the local coordinates of the thread position in the tile it is working on
    int x = tile_start_x + threadIdx.x * 2;
    int y = 0;

    // Each thread updates 2 diagonals at once
    double cov1, cov2, cov3, cov4;
    
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
        initialize_tile_memory<DTYPE, BLOCKSZ, tile_height, tile_width>(profile, df, dg, norms, local_mp_col, local_mp_row,
                                                                        df_col, df_row, dg_col, dg_row, inorm_col, inorm_row,
                                                                        n, tile_start_x, tile_start_y);
        // Start of new tile, sync
        __syncthreads();

        // There are 2 pathways here, most of the time we take the fast path (top), at the very end of the kernel we take the slower path (bottom)
        if(x + tile_height < n) {
            for(int i = 0, j = threadIdx.x * 2; i < tile_height; i+=2, j+=2) {
                do_iteration_unroll_2(i,j,x + i,y + i,n, cov1,cov2, df_col, df_row, dg_col, dg_row, inorm_col, inorm_row, local_mp_col, local_mp_row);
            }
		    x += tile_height;
            y += tile_height;
        } else {
            int localX = threadIdx.x * 2;
            int localY = 0;
            while(x + 2 < n && y + 1 < n) {
                do_iteration_unroll_2(localY,localX,x +localY,y + localY,n, cov1,cov2, df_col, df_row, dg_col, dg_row, inorm_col, inorm_row, local_mp_col, local_mp_row);
                //do_iteration_unroll_2_check(localY,localX,x,y,n,qt,qt2,A_low, A_high, B_low, B_high, mean_x, mean_y, inv_std_x, inv_std_y, localMPMain, localMPOther); 
                x += 2;
                y += 2;
                localX += 2;
                localY += 2;
            }
	    }
        
        // After this sync, the caches will be updated with the best so far values for this tile
        __syncthreads();

        // If we updated any values in the cached MP, try to push them to the global "master" MP
        if (tile_start_x + threadIdx.x < n) {
        	MPatomicMax(profile + tile_start_x + threadIdx.x, local_mp_col[threadIdx.x].floats[0], local_mp_col[threadIdx.x].ints[1]);
        }
        if (tile_start_x + threadIdx.x + BLOCKSZ < n) {
        	MPatomicMax(profile + BLOCKSZ + tile_start_x + threadIdx.x, local_mp_col[threadIdx.x + BLOCKSZ].floats[0], local_mp_col[threadIdx.x + BLOCKSZ].ints[1]);
        }
        if (tile_start_x + threadIdx.x + BLOCKSZ * 2 < n && threadIdx.x < tile_height) {
        	MPatomicMax(profile + BLOCKSZ * 2 + tile_start_x + threadIdx.x, local_mp_col[threadIdx.x + BLOCKSZ * 2].floats[0], local_mp_col[threadIdx.x + BLOCKSZ * 2].ints[1]);
        }
        if (tile_start_y + threadIdx.x < n && threadIdx.x < tile_height) {
        	MPatomicMax(profile + tile_start_y + threadIdx.x, local_mp_row[threadIdx.x].floats[0], local_mp_row[threadIdx.x].ints[1]);
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
        profile[tid] = sqrt(max(2*(m - profile[tid]), 0.0));
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
    int num_workers = ceil(ceil((n - (m / 4)) / (float) devices.size()) / 2.0);
    
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
        compute_statistics<DTYPE>(T_dev[device], norms[device], df[device], dg[device], means[device], n, m, streams.at(device));
        sliding_dot_products_and_distance_profile<DTYPE, CUFFT_DTYPE>(T_dev[device], T_dev[device], means[device], QT_dev[device], T_h.size(), m, streams.at(device));
        
        thrust::device_ptr<unsigned long long int> ptr = thrust::device_pointer_cast(profile_merged[device]);
        thrust::transform(thrust::cuda::par.on(streams.at(device)), profile_dev[device], profile_dev[device] + n, profile_idx_dev[device], profile_merged[device], combiner);
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        gpuErrchk(cudaPeekAtLastError());
        printf("Start main kernel on GPU %d\n", device);

        cudaEventRecord(clocks_start[device], streams.at(device));
        WavefrontUpdateSelfJoin<float, WORK_SIZE, AMT_UNROLL><<<dim3(ceil(num_workers / (double) WORK_SIZE), 1, 1),dim3(WORK_SIZE, 1,1), 0, streams.at(device)>>>(QT_dev[device], df[device], dg[device], norms[device], profile_merged[device], m, n, count, devices.size());
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
    //cross_correlation_to_ed<<<dim3(ceil(n / (float) WORK_SIZE), 1, 1), dim3(WORK_SIZE, 1, 1)>>>(profile_dev[devices.at(0)], n, m); 
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



