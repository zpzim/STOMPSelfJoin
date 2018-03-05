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
static const unsigned int AMT_UNROLL = 4;

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
void compute_statistics(const DTYPE *T, DTYPE *means, DTYPE *stds, size_t n, size_t m, cudaStream_t s)
{
    square<DTYPE> sqr;
    dim3 grid(ceil(n / (double) WORK_SIZE), 1,1);
    dim3 block(WORK_SIZE, 1, 1);
    
    DTYPE *scratch;
    cudaMalloc(&scratch, sizeof(DTYPE) * n);
    gpuErrchk(cudaPeekAtLastError());
    
    thrust::device_ptr<const DTYPE> dev_ptr_T = thrust::device_pointer_cast(T);
    thrust::device_ptr<DTYPE> dev_ptr_scratch = thrust::device_pointer_cast(scratch);

    // Compute prefix sum in scratch
    thrust::inclusive_scan(thrust::cuda::par.on(s), dev_ptr_T, dev_ptr_T + n + m - 1, dev_ptr_scratch, thrust::plus<DTYPE>());
    gpuErrchk(cudaPeekAtLastError());
    // Use prefix sum to compute sliding mean
    sliding_mean<DTYPE><<<grid, block, 0, s>>>(scratch, m, n, means);
    gpuErrchk(cudaPeekAtLastError());
    // Compute prefix sum of squares in scratch
    thrust::transform_inclusive_scan(thrust::cuda::par.on(s), dev_ptr_T, dev_ptr_T + n + m - 1, dev_ptr_scratch, sqr,thrust::plus<DTYPE>());
    gpuErrchk(cudaPeekAtLastError());
    // Use prefix sum of squares to compute the sliding standard deviation
    sliding_std<DTYPE><<<grid, block, 0, s>>>(scratch, m, n, means, stds);
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
__global__ void populate_reverse_pad(const DTYPE *Q, DTYPE *Q_reverse_pad, const int window_size, const int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < window_size) {
        Q_reverse_pad[tid] = Q[window_size - 1 - tid];
    }else if(tid < size){ 
        Q_reverse_pad[tid] = 0;
    }
}

template<class DTYPE, class CUFFT_DTYPE>
void sliding_dot_products_and_distance_profile(DTYPE* T, DTYPE* Q, DTYPE *QT, const int size, const int window_len, cudaStream_t s)
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
    populate_reverse_pad<DTYPE><<<dim3(ceil(size / (float) WORK_SIZE),1,1), block, 0, s>>>(Q, Q_reverse_pad, window_len, size);
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
__device__ inline void MPatomicMax(volatile unsigned long long int* address, float val, unsigned int idx)
{
    mp_entry loc, loctest;
    loc.floats[0] = val;
    loc.ints[1] = idx;
    loctest.ulong = *address;
    while (loctest.floats[0] < val){
        loctest.ulong = atomicCAS((unsigned long long int*) address, loctest.ulong,  loc.ulong);
    }
}

template<unsigned int tile_height>
//Updates the global matrix profile based on a block-local, cached version
__device__ inline void UpdateMPGlobalMax(unsigned long long* profile, volatile mp_entry* localMP, const int chunk, const int offset, const int n){
    
    int x = chunk*(tile_height)+threadIdx.x;
    if(x < n && ((mp_entry*) profile)[x].floats[0] < localMP[threadIdx.x+offset].floats[0])
    {
            MPatomicMax(&profile[x], localMP[threadIdx.x+offset].floats[0], localMP[threadIdx.x+offset].ints[1]);
    }
}

template<class DTYPE, unsigned int BLOCKSZ, unsigned int tile_height>
__device__ inline void initialize_tile_memory(const unsigned long long int *profile, const double *T,
                                              const double *means, const double *inv_stds,
                                              volatile mp_entry localMPMain[], volatile mp_entry localMPOther[],
                                              DTYPE A_low[], DTYPE A_high[], DTYPE B_low[], DTYPE B_high[],
                                              DTYPE mean_x[], DTYPE mean_y[], DTYPE inv_std_x[],
                                              DTYPE inv_std_y[], const unsigned int n, const unsigned int m,
                                              const unsigned int mainStart, const unsigned int otherStart,
                                              const unsigned int x, const unsigned int y)
{
    // Update local cache to point to the next chunk of the MP
    // We may not get the 'freshest' values from the global array, but it doesn't really matter too much
    if (mainStart + threadIdx.x < n) {
        localMPMain[threadIdx.x].ulong = profile[mainStart + threadIdx.x];
    } else {
        localMPMain[threadIdx.x].floats[0] = CC_MIN;
        localMPMain[threadIdx.x].ints[1] = 0;
    }

    // Each thread grabs 2 values for the main cache
    if (threadIdx.x < tile_height && mainStart+threadIdx.x+BLOCKSZ < n) {
        localMPMain[BLOCKSZ + threadIdx.x].ulong = profile[mainStart + BLOCKSZ + threadIdx.x];
    } else if (threadIdx.x < tile_height) {
        localMPMain[threadIdx.x + BLOCKSZ].floats[0] = CC_MIN;
        localMPMain[threadIdx.x + BLOCKSZ].ints[1] = 0;
    }
    
    // We also update the cache for the transposed tile
    if (threadIdx.x < tile_height && otherStart+threadIdx.x < n) {
        localMPOther[threadIdx.x].ulong = profile[otherStart + threadIdx.x];
    } else if (threadIdx.x < tile_height) {
        localMPOther[threadIdx.x].floats[0] = CC_MIN;
        localMPOther[threadIdx.x].ints[1] = 0;
    }

    // Update the other cached values to reflect the upcoming tile
    if (x <  n + m - 1) {
        A_low[threadIdx.x] = T[x];
    }
    if (threadIdx.x < tile_height && x + BLOCKSZ < n + m - 1) {
        A_low[threadIdx.x + BLOCKSZ] = T[x + BLOCKSZ];
    }
    
    if (x + m < n + m - 1) {
        A_high[threadIdx.x] = T[x + m];
    }
    if (threadIdx.x < tile_height && x + BLOCKSZ + m < n + m - 1) {
        A_high[threadIdx.x + BLOCKSZ] = T[x + BLOCKSZ + m];
    }
    if (threadIdx.x < tile_height && y + threadIdx.x < n + m - 1) {
        B_low[threadIdx.x] = T[y + threadIdx.x];
    }
    if (threadIdx.x < tile_height && y + threadIdx.x + m < n + m - 1) {
        B_high[threadIdx.x] = T[y + threadIdx.x + m];
    }
    if (x < n) {
        inv_std_x[threadIdx.x] = inv_stds[x];
        // We precompute part of the distance calculation in the mean_x variable
        // This saves us a multiply in the main loop
        mean_x[threadIdx.x] = means[x] * m;
    }
    if (threadIdx.x < tile_height && x + BLOCKSZ < n) {
        inv_std_x[threadIdx.x + BLOCKSZ] = inv_stds[x + BLOCKSZ];
        // We precompute part of the distance calculation in the mean_x variable
        // This saves us a multiply in the main loop
        mean_x[threadIdx.x + BLOCKSZ] = means[x + BLOCKSZ] * m;
    }
    if (threadIdx.x < tile_height && y + threadIdx.x < n) {
        inv_std_y[threadIdx.x] = inv_stds[y + threadIdx.x];
        mean_y[threadIdx.x] = means[y + threadIdx.x];
    }
}

//Computes the matrix profile given the sliding dot products for the first query and the precomputed data statisics
template<class DTYPE, unsigned int BLOCKSZ, unsigned int UNROLL_COUNT>
__global__ void WavefrontUpdateSelfJoinMaxSharedMem(const double* QT, const double* T, const double* inv_stds, const double* means, unsigned long long int* profile, unsigned int m, unsigned int n, int startPos, int numDevices, struct reg_mem<UNROLL_COUNT> mem)
{
    //Factor and threads per block must both be powers of two where: factor <= threads per block
    // UNROLL_COUNT * factor must also evenly divide WORK_SIZE
    // The values that are set here should give good performance already
    // but may be fine tuned for other Nvidia architectures
    //Use the smallest power of 2 possible for your GPU
    const int factor = 4;
    const int tile_height = BLOCKSZ / factor;
    const int tile_width = tile_height + BLOCKSZ;
    __shared__ mp_entry localMPMain[tile_width];
    __shared__ mp_entry localMPOther[tile_height];
    __shared__ DTYPE A_low[tile_width];
    __shared__ DTYPE A_high[tile_width];
    __shared__ DTYPE inv_std_x[tile_width];
    __shared__ DTYPE inv_std_y[tile_height];
    __shared__ DTYPE mean_x[tile_width];
    __shared__ DTYPE mean_y[tile_height];
    __shared__ DTYPE B_high[tile_height];
    __shared__ DTYPE B_low[tile_height];


    int exclusion = (m / 4);
    int a = ((blockIdx.x * numDevices) + startPos) * BLOCKSZ + threadIdx.x;
    double qt_curr;
    int localX = threadIdx.x;
    int localY = 0;
    int chunkIdxMain = (a / BLOCKSZ) * factor;
    int chunkIdxOther = 0;
    int mainStart = tile_height * chunkIdxMain;
    int otherStart = 0;
    if (a < n) {
        mem.qt[0] = QT[a];
    }
    
    // x is the global column of the distance matrix
    // y is the global row of the distance matrix
    // Each thread starts on the first row and works its way to the down-right diagonal
    int x = a;
    int y = 0;

    // The first threads are acutally computing the trivial match between the same subsequence
    // we exclude these from the calculation
    bool excluded;
    if (x <= exclusion) {
        excluded = true;
    } else {
        excluded = false;
    }
    // Initialize the first tile's shared memory 
    initialize_tile_memory<DTYPE, BLOCKSZ, tile_height>(profile, T, means, inv_stds, localMPMain, localMPOther,
                                            A_low, A_high, B_low, B_high, mean_x, mean_y, inv_std_x,
                                            inv_std_y, n, m, mainStart, otherStart, x, y);

    /////////////////////////////////////    
    // Main loop
    /////////////////////////////////////
    // Each threadblock finds all the distances on a 'metadiagonal'
    // We use a tiled approach for each thread block
    // The tiles are horizontal slices of the diagonal, think of a parallelogram cut
    // from a diagonal slice of the distance matrix 
    while (mainStart < n && otherStart < n)
    {
        // Start of new tile, sync
        __syncthreads();

        // The first 'm/4' diagonals are computing the exclusion zone, so we don't want to
        // include their distances in the calculation
        if (excluded) {
                x += tile_height;
                y += tile_height;
        } else {
            // Process the tile
            // Each iteration generates the next 4 distances
            // This loop is partially unrolled to improve instruction level parallelism
            while (x < n - UNROLL_COUNT + 1 && localY < tile_height)
            {
                // Update the QT value for the next iteration
                #pragma unroll
                for (int i = 0; i < UNROLL_COUNT - 1; ++i) {
                    mem.qt[i + 1] = mem.qt[i] - A_low[localX + i] * B_low[localY + i] + A_high[localX + i] * B_high[localY + i];
                }
                
                // Compute the next partial distance value
                // We defer some of the calculation until after the kernel has finished, this saves us several
                // long latency math operations in this critical path.
                // The distance computed here can be converted to the true z-normalized euclidan
                // distance in constant time
                // mean_x has already been multiplied with the window size 'm' when the tile was populated
                #pragma unroll
                for (int i = 0; i < UNROLL_COUNT; ++i) {
                    mem.dist[i] = (static_cast<float>(mem.qt[i]) - (mean_x[localX + i] * mean_y[localY + i])) * inv_std_x[localX + i] * inv_std_y[localY + i];
                }


                mem.qt[0] = mem.qt[UNROLL_COUNT - 1] - A_low[localX + UNROLL_COUNT - 1] * B_low[localY + UNROLL_COUNT - 1] + A_high[localX + UNROLL_COUNT - 1] * B_high[localY + UNROLL_COUNT - 1];

                // Update the cache with the new max value atomically
                #pragma unroll
                for (int i = 0; i < UNROLL_COUNT; ++i) {
                    MPatomicMax((unsigned long long int*) (localMPMain + localX + i), mem.dist[i], y + i);
                    MPatomicMax((unsigned long long int*) (localMPOther + localY + i), mem.dist[i], x + i);
                }

                x += UNROLL_COUNT;
                y += UNROLL_COUNT;
                localX += UNROLL_COUNT;
                localY += UNROLL_COUNT;
            }

            qt_curr = mem.qt[0];

            // Finish the remaining iterations of the tile
            while (x < n && localY < tile_height) {
                float dist = (static_cast<float>(qt_curr) - (mean_x[localX] * mean_y[localY])) * inv_std_x[localX] * inv_std_y[localY];
                qt_curr = qt_curr - A_low[localX] * B_low[localY] + A_high[localX] * B_high[localY];
                MPatomicMax((unsigned long long int*) (localMPMain + localX), dist, y);
                MPatomicMax((unsigned long long int*) (localMPOther + localY), dist, x);

                x++;
                y++;
                localX++;
                localY++;
            }

        }

        // After this sync, the caches will be updated with the best so far values for this tile
        __syncthreads();

        // If we updated any values in the cached MP, try to push them to the global "master" MP
        UpdateMPGlobalMax<tile_height>(profile, localMPMain, chunkIdxMain, 0, n);
        if (threadIdx.x < tile_height) {
            UpdateMPGlobalMax<tile_height>(profile, localMPMain, chunkIdxMain + factor, BLOCKSZ, n);
            UpdateMPGlobalMax<tile_height>(profile, localMPOther, chunkIdxOther, 0, n);
        }


        // Update the tile position
        mainStart += tile_height;
        otherStart += tile_height;

        // Make sure our updates were committed before we pull in the next tile
        __threadfence_block();
    
        // Initialize the next tile's shared memory 
        initialize_tile_memory<DTYPE, BLOCKSZ, tile_height>(profile, T, means, inv_stds, localMPMain, localMPOther,
                                                A_low, A_high, B_low, B_high, mean_x, mean_y, inv_std_x,
                                                inv_std_y, n, m, mainStart, otherStart, x, y);

        // Reset the tile local positions
        localY = 0;
        localX = threadIdx.x;
        chunkIdxMain++;
        chunkIdxOther++;    
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
    
    unordered_map<int, DTYPE*> T_dev, QT_dev, means, stds;
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
        stds.insert(make_pair(device, (DTYPE*) 0));
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
        cudaMalloc(&stds.at(device), sizeof(DTYPE) * profile_h.size());
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
    int num_workers = ceil(n / (float) devices.size());
    
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
        compute_statistics<DTYPE>(T_dev[device], means[device], stds[device], n, m, streams.at(device));
        sliding_dot_products_and_distance_profile<DTYPE, CUFFT_DTYPE>(T_dev[device], T_dev[device], QT_dev[device], T_h.size(), m, streams.at(device));
        
        thrust::device_ptr<unsigned long long int> ptr = thrust::device_pointer_cast(profile_merged[device]);
        thrust::transform(thrust::cuda::par.on(streams.at(device)), profile_dev[device], profile_dev[device] + n, profile_idx_dev[device], profile_merged[device], combiner);
        printf("Start main kernel on GPU %d\n", device);
        //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        cudaEventRecord(clocks_start[device], streams.at(device));
        WavefrontUpdateSelfJoinMaxSharedMem<float, WORK_SIZE, AMT_UNROLL><<<dim3(ceil(num_workers / (double) WORK_SIZE), 1, 1),dim3(WORK_SIZE, 1,1), 0, streams.at(device)>>>(QT_dev[device], T_dev[device], stds[device], means[device], profile_merged[device], m, n, count, devices.size(), reg_mem<AMT_UNROLL>());
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

    printf("Finished STOMP to generate partial matrix profile of size %lu on %d devices:\n", n, devices.size());

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
        cudaFree(stds[device]);
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



