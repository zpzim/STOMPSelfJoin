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
#include <pthread.h>
#include <math.h>

#include "cuda_profiler_api.h"
#include "STOMP.h"

const char * format_str = "%lf";
const char * format_str_n = "%lf\n";

time_t START;

struct thread_args{
	unsigned int tid;
	thrust::device_vector<DATA_TYPE> *Ta, *Tb;
	thrust::device_vector<unsigned long long int> *profile;
	thrust::device_vector<unsigned int> *profileIdxs;
	unsigned int m;
	int exclusion;
	int maxJoin;
	int start, end;
	int numWorkers;
};

struct thread_args targs[NUM_THREADS];
int nDevices;
pthread_t threads[NUM_THREADS];
static const unsigned int WORK_SIZE = 1024;
cufftHandle plan[NUM_THREADS], plan2[NUM_THREADS], plan3[NUM_THREADS], plan4[NUM_THREADS];


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



void STOMPinit(int size){
	for(int i = 0; i < NUM_THREADS; ++i){
	    printf("HERE\n");
	    gpuErrchk(cudaSetDevice(i % nDevices));
		cufftPlan1d(&plan[i], size * 2, CUFFT_FORWARD_PLAN, 1);
		cufftPlan1d(&plan2[i], size * 2, CUFFT_REVERSE_PLAN, 1);
		printf("HERE\n");
	}
	gpuErrchk(cudaSetDevice(0));
}

void STOMPclean(int size){
	for(int i = 0; i < NUM_THREADS; ++i){
	    cufftDestroy(plan[i]);
	    cufftDestroy(plan2[i]);
		cufftPlan1d(&plan[i], size * 2, CUFFT_FORWARD_PLAN, 1);
		cufftPlan1d(&plan2[i], size * 2, CUFFT_REVERSE_PLAN, 1);
	}
}

//Reads input time series from file
void readFile(const char* filename, thrust::host_vector<DATA_TYPE>& v){
	FILE* f = fopen( filename, "r");
	DATA_TYPE num;
	while(!feof(f)){
			fscanf(f, format_str, &num);
			v.push_back(num);
		}
	v.pop_back();
	fclose(f);
}





//This kernel computes a sliding mean with specified window size and a corresponding prefix sum array (A)
__global__ void slidingMean(DATA_TYPE* A,  int window, unsigned int size, DATA_TYPE* Means){
	__const__ DATA_TYPE coeff = 1.0 / (DATA_TYPE)window;
	int a = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.x * blockDim.x + threadIdx.x + window;

	if(a == 0){
		Means[a] = A[window - 1] * coeff;
	}
	if(a < size - 1){
		//printf("%d\n", a + 1);
		Means[a + 1] = (A[b] - A[a]) * coeff;
	}
}

//This kernel computes a sliding standard deviaiton with specified window size, the corresponding means of each element, and the prefix squared sum at each element
__global__ void slidingStd(DATA_TYPE* squares, unsigned int window, unsigned int size, DATA_TYPE* Means, DATA_TYPE* stds){
	__const__ DATA_TYPE coeff = 1 / (DATA_TYPE)window;
	int a = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.x * blockDim.x + threadIdx.x + window;
	if(a == 0){
		stds[a] = sqrt((squares[window - 1] * coeff) - (Means[a] * Means[a]));
	}
	else if(b < size + window)
		stds[a] = sqrt(((squares[b - 1] - squares[a - 1]) * coeff) - (Means[a] * Means[a]));
}

//This kernel computes the distance profile for a given window position, as long as the index is outside the exclusionZone
__global__ void CalculateDistProfile(DATA_TYPE* QT, DATA_TYPE* D, DATA_TYPE* Means, DATA_TYPE* stds, int m, int start, int n){
	const DATA_TYPE Qmean = Means[start];
	const DATA_TYPE Qstd = stds[start];
	const int exclusionZone =  m / 4;
	int a = blockIdx.x * blockDim.x + threadIdx.x;
	if(a < n && a > start - exclusionZone && a < start + exclusionZone ){
	//if(a == start){
		D[a] = FLT_MAX;
	}else if( a < n){
	    //D[a] = sqrt(abs(2 * (m - (QT[a] - m * Means[a] * Qmean) / (stds[a] * Qstd))));
		D[a] = sqrt(abs(2 * (m - (QT[a] - m * Means[a] * Qmean) / (stds[a] * Qstd))));
	}
}

//This kernel divides each element in A by divisor
__global__ void divideBy(DATA_TYPE* A, DATA_TYPE divisor, unsigned int size){
	int a = blockIdx.x * blockDim.x + threadIdx.x;
	if(a < size){
		A[a] /= divisor;
	}
}

//Computes the sliding dot products for a given query using FFT
__host__ void SlidingDotProducts(const thrust::device_vector<DATA_TYPE>& Q, const thrust::device_vector<DATA_TYPE>& T, 	thrust::device_vector<DATA_TYPE>&  P, cufftHandle plan, cufftHandle plan2){		
	int sz = T.size() * 2;
	printf("Starting FFT Forward 1\n");
	thrust::device_vector<__CUFFT_TYPE__> Qrc(sz);
	gpuErrchk( cudaPeekAtLastError() );
	thrust::device_vector<DATA_TYPE> Qr(sz);
	gpuErrchk( cudaPeekAtLastError() );
	thrust::reverse_copy(Q.begin(), Q.end(), Qr.begin());
	gpuErrchk( cudaPeekAtLastError() );
	time_t start, now;
	time(&start);
	CUFFT_FORWARD__(plan, Qr.data().get(), Qrc.data().get());
	gpuErrchk( cudaPeekAtLastError() );
	time(&now);
	printf("FFT Forward 1 took %f seconds\n", difftime(start, now));
	Qr.clear();
	Qr.shrink_to_fit();
	thrust::host_vector<__CUFFT_TYPE__> Qrc_h = Qrc;
	Qrc.clear();
	Qrc.shrink_to_fit();
	
	printf("Allocating Tac\n");
	thrust::device_vector<__CUFFT_TYPE__> Tac(sz);
	printf("Allocating Ta\n");
	thrust::device_vector<DATA_TYPE> Ta(sz);
	thrust::copy(T.begin(), T.end(), Ta.begin());
	gpuErrchk( cudaPeekAtLastError() );
	
	time(&start);
	CUFFT_FORWARD__(plan, Ta.data().get(), Tac.data().get());
	gpuErrchk( cudaPeekAtLastError() );
	time(&now);
	printf("FFT Forward 2 took %f seconds\n", difftime(start, now));
	
	
	Ta.clear();
	Ta.shrink_to_fit();
	Qrc = Qrc_h;
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(Qrc.begin(), Tac.begin())),
							thrust::make_zip_iterator(thrust::make_tuple(Qrc.end(), Tac.end())),
							multiply());
	printf("Finished elementwise multiply\n");
	Tac.clear();
	Tac.shrink_to_fit();
	P.resize(sz);
    	printf("Starting FFT reverse\n");
    	time(&start);
	CUFFT_REVERSE__(plan2, Qrc.data().get(), P.data().get());
	gpuErrchk( cudaPeekAtLastError() );
	time(&now);
	printf("FFT Reverse took %f seconds\n", difftime(start, now));
	
	dim3 grid(sz / WORK_SIZE + 1, 1, 1);
	dim3 block(WORK_SIZE, 1, 1);
	printf("%d\n",sz / WORK_SIZE);
	divideBy<<<grid,block>>>(P.data().get(), P.size(), P.size());
	gpuErrchk( cudaPeekAtLastError() );

}

//Atomically updates the MP/idxs using a single 64-bit integer. We lose a small amount of precision in the output, if we do not do this we are unable
// to atomically update both the matrix profile and the indexes without using a critical section and dedicated locks.
__device__ inline unsigned long long int MPatomicMin(volatile unsigned long long int* address, double val, unsigned int idx)
{
	float fval = (float)val;
	mp_entry loc, loctest;
	loc.floats[0] = fval;
	loc.ints[1] = idx;
	loctest.ulong = *address;
	while (loctest.floats[0] > fval){
		loctest.ulong = atomicCAS((unsigned long long int*) address, loctest.ulong,  loc.ulong);
	}
	return loctest.ulong;
}

//Updates the global matrix profile based on a block-local, cached version
__device__ inline void UpdateMPGlobal(volatile unsigned long long* profile, volatile mp_entry* localMP, const int chunk, const int offset, const int n){
	
	int x = chunk*blockDim.x+threadIdx.x;
	if(x < n && ((mp_entry*) profile)[x].floats[0] > localMP[threadIdx.x+offset].floats[0])
	{
			MPatomicMin(&profile[x], localMP[threadIdx.x+offset].floats[0], localMP[threadIdx.x+offset].ints[1]);
	}
}

//Computes the matrix profile given the sliding dot products for the first query and the precomputed data statisics
__global__ void WavefrontUpdateSelfJoin(double* QT, double* Ta, double* Tb, double* means, double* stds, volatile unsigned long long int* profile, unsigned int m, unsigned int n, int startPos, int endPos, int numDevices){
	__shared__ volatile mp_entry localMPMain[WORK_SIZE * 2];
	__shared__ volatile mp_entry localMPOther[WORK_SIZE];
	__shared__ volatile bool updated[3];


	int a = ((blockIdx.x * numDevices) + startPos) * blockDim.x + threadIdx.x;
	//const int b = ((blockIdx.x * numDevices) + startPos + 1) * blockDim.x;
	int exclusion = m / 4;
	double workspace;
	int localX = threadIdx.x + 1;
	int localY = 1;
	int chunkIdxMain = a / blockDim.x;
	int chunkIdxOther = 0;
	int mainStart = blockDim.x * chunkIdxMain;
	int otherStart = 0;
	if(a < n){
		workspace = QT[a];
	}else{
		workspace = -1;
	}
	//Initialize Shared Data
	if(mainStart+threadIdx.x < n){
		localMPMain[threadIdx.x].ulong = profile[mainStart + threadIdx.x];
	}else{
		localMPMain[threadIdx.x].floats[0] = FLT_MAX;
		localMPMain[threadIdx.x].ints[1] = 0;
	}
	if(mainStart+threadIdx.x+blockDim.x < n){
		localMPMain[blockDim.x + threadIdx.x].ulong = profile[mainStart + blockDim.x + threadIdx.x];
	}else{
		localMPMain[blockDim.x + threadIdx.x].floats[0] = FLT_MAX;
		localMPMain[blockDim.x + threadIdx.x].ints[1] = 0;
	}
	if(otherStart+threadIdx.x < n){
		localMPOther[threadIdx.x].ulong = profile[otherStart + threadIdx.x];
	}else{
		localMPOther[threadIdx.x].floats[0] = FLT_MAX;
		localMPOther[threadIdx.x].ints[1] = 0;
	}
	if(threadIdx.x == 0)
	{
		updated[0] = false;
		updated[1] = false;
		updated[2] = false;
	}
	int x = a + 1;
	int y = 1;
	
	while(mainStart < n && otherStart < n)
	{
		__syncthreads();
		//Update to the end of the current chunk
		while(x < n && y < n && localY < blockDim.x)
		{
			workspace = workspace - Ta[x - 1] * Tb[y - 1] + Ta[x + m - 1] * Tb[ y + m - 1];
			if(!(x > y - exclusion && x < y + exclusion))
			{
				//Compute the next distance value
				double dist = sqrt(abs(2 * (m - (workspace - m * means[x] * means[y]) / (stds[x] * stds[y]))));

				//Check cache to see if we even need to try to update
				if(localMPMain[localX].floats[0] > dist)
				{	
					//Update the cache with the new min value atomically
					MPatomicMin((unsigned long long int*)&localMPMain[localX], dist, y);
					if(localX < blockDim.x && !updated[0]){
						updated[0] = true;
					}else if(!updated[1]){
						updated[1] = true;
					}
				}
				//Check cache to see if we even need to try to update
				if(localMPOther[localY].floats[0] > dist)
				{
					//Update the cache with the new min value atomically
					MPatomicMin((unsigned long long int*)&localMPOther[localY], dist, x);
					if(!updated[2]){
						updated[2] = true;
					}				
				}
			}			
			++x;
			++y;
			++localX;
			++localY;
		}
		__syncthreads();
		//If we updated any values in the cached MP, try to push them to the global "master" MP
		if(updated[0]){
			UpdateMPGlobal(profile, localMPMain, chunkIdxMain, 0,n);
		}
		if(updated[1]){
			UpdateMPGlobal(profile, localMPMain, chunkIdxMain + 1, blockDim.x,n);
		}
		if(updated[2]){
			UpdateMPGlobal(profile, localMPOther, chunkIdxOther, 0,n);	
		}
		__syncthreads();	
		if(threadIdx.x == 0){
			updated[0] = false;
			updated[1] = false;
			updated[2] = false;
		}		
		mainStart += blockDim.x;
		otherStart += blockDim.x;
		//Update local cache to point to the next chunk of the MP
		if(mainStart+threadIdx.x < n)
		{
			localMPMain[threadIdx.x].ulong = profile[mainStart + threadIdx.x];
		}
		else
		{
			localMPMain[threadIdx.x].floats[0] = FLT_MAX;
			localMPMain[threadIdx.x].ints[1] = 0;
		}
		if(mainStart+threadIdx.x+blockDim.x < n)
		{
			localMPMain[blockDim.x + threadIdx.x].ulong = profile[mainStart + blockDim.x + threadIdx.x];
		}
		else
		{
			localMPMain[threadIdx.x + blockDim.x].floats[0] = FLT_MAX;
			localMPMain[threadIdx.x + blockDim.x].ints[1] = 0;
		}
		if(otherStart+threadIdx.x < n)
		{
			localMPOther[threadIdx.x].ulong = profile[otherStart + threadIdx.x];
		}
		else
		{
			localMPOther[threadIdx.x].floats[0] = FLT_MAX;
			localMPOther[threadIdx.x].ints[1] = 0;
		}	
		localY = 0;
		localX = threadIdx.x;
		chunkIdxMain++;
		chunkIdxOther++;	
	}
}

//Performs STOMP algorithm
void* doThreadSTOMP(void* argsp){
    
    	thread_args* args = (thread_args*) argsp;
    	int tid = args->tid;
    	gpuErrchk(cudaSetDevice(tid % nDevices));
	thrust::device_vector<DATA_TYPE>* Ta = args -> Ta;
 	thrust::device_vector<DATA_TYPE>* Tb = args -> Tb;
	thrust::device_vector<unsigned long long int>* profile = args -> profile;
	thrust::device_vector<unsigned int>* profileIdxs = args -> profileIdxs;
	int numWorkers = args ->numWorkers;
	int m = args -> m;

	unsigned int start = args -> start;
	unsigned int end = args -> end;

	unsigned int n = Ta ->size() - m + 1;
	unsigned int n2 = Tb -> size() - m + 1;
	unsigned int sz = Ta -> size();

    	thrust::plus<DATA_TYPE> op1;
	square op2;
	printf("allocating grids\n");
	dim3 grid(Ta -> size() / WORK_SIZE + 1, 1, 1);
	dim3 grid3(n / WORK_SIZE + 1, 1, 1);
   	dim3 block(WORK_SIZE, 1, 1);
	
	double oldTime = 0;

	thrust::device_vector<DATA_TYPE> Qb(m);
	gpuErrchk( cudaPeekAtLastError() );
    	thrust::copy(thrust::cuda::par,Tb -> begin(), Tb -> begin() + m, Qb.begin());
    	gpuErrchk( cudaPeekAtLastError() );
	
	thrust::device_vector<DATA_TYPE> QT;
		
	SlidingDotProducts(Qb, *Ta, QT, plan[tid], plan2[tid]);

	thrust::device_vector<DATA_TYPE> QTtrunc(n);
	gpuErrchk( cudaPeekAtLastError() );
	thrust::copy(thrust::cuda::par,QT.begin() + m - 1, QT.begin() + m + n - 1, QTtrunc.begin());
	gpuErrchk( cudaPeekAtLastError() );
	QT.clear();
	QT.shrink_to_fit();
    
	thrust::device_vector<DATA_TYPE> Means(n), stds(n), squares(Ta -> size()), sums(Ta -> size());
	thrust::inclusive_scan(Ta -> begin(),Ta -> end(),sums.begin(), op1);
	thrust::transform_inclusive_scan(Ta -> begin(), Ta -> end(), squares.begin(), op2,op1);
	slidingMean<<<grid, block>>>(sums.data().get(),m, n, Means.data().get());
	gpuErrchk( cudaPeekAtLastError() );
	slidingStd<<<grid, block>>>(squares.data().get(), m, n, Means.data().get(), stds.data().get());
	gpuErrchk( cudaPeekAtLastError() );
	
	sums.clear();
	squares.clear();
	sums.shrink_to_fit();
	squares.shrink_to_fit();
	
    	printf("allocating DP");
    	thrust::device_vector<DATA_TYPE> D;
    	D.resize(n,_MAX_VAL_);
	
	CalculateDistProfile<<<grid, block>>>(QTtrunc.data().get(), D.data().get(), Means.data().get(), stds.data().get(), m, 0, n);
	gpuErrchk( cudaPeekAtLastError() );

    	//Initialize the indexes to the starting position
	profileIdxs -> resize(n,1);
	profile->resize(n, 0);

	thrust::device_vector<double>::iterator it = thrust::min_element(D.begin(),D.end());
	unsigned int pos = it - D.begin();
	double val = *it;
	//cout << pos << " " << val; 
	(*profileIdxs)[0] = pos;
	D[0] = *it;
	
	thrust::transform(D.begin(), D.end(), profileIdxs->begin(), profile->begin(), MPIDXCombine());
	D.clear();
	D.shrink_to_fit();

	time_t start2, now2;
	time_t lastLogged;
	time(&start2);
	time(&lastLogged);

	WavefrontUpdateSelfJoin<<<dim3(ceil(numWorkers / (double) WORK_SIZE), 1, 1),dim3(WORK_SIZE, 1,1)>>>(QTtrunc.data().get(), Ta -> data().get(), Tb -> data().get(), Means.data().get(), stds.data().get(), profile -> data().get(), m, n, start, end, NUM_THREADS);
	gpuErrchk( cudaPeekAtLastError() );	
	cudaDeviceSynchronize();
	//std::cout << thrust::reduce(counts.begin(), counts.end(), 0, thrust::plus<unsigned long long>()) << std::endl;	
	time_t now3;
	time(&now3);
	printf("Finished thread %d over all iterations in %lf seconds\n", tid, difftime(now3, start2) + oldTime);
	pthread_exit(0);
}


//Allocates threads on a CPU to distribute work to each specified device
__host__ void STOMP(thrust::host_vector<DATA_TYPE>& Ta, unsigned int m,
		    thrust::host_vector<float>& profile_h, thrust::host_vector<unsigned int>& profileIdxs_h){
	
	gpuErrchk(cudaGetDeviceCount(&nDevices));
	STOMPinit(Ta.size());
	thrust::device_vector<DATA_TYPE>* Ta_d = new thrust::device_vector<DATA_TYPE>[nDevices];
	thrust::device_vector<unsigned long long int>* Profs[NUM_THREADS];
	thrust::device_vector<unsigned int>* ProfsIdxs[NUM_THREADS];
	for(int i = 0; i < nDevices; ++i){
	    gpuErrchk(cudaSetDevice(i));
	    Ta_d[i] = Ta; 
	}
	for(int i = 0; i < NUM_THREADS; ++i){
	    gpuErrchk(cudaSetDevice(i % nDevices));
	    Profs[i] = new thrust::device_vector<unsigned long long int>();
	    ProfsIdxs[i] = new thrust::device_vector<unsigned int>();
	}
	gpuErrchk(cudaSetDevice(0));
	unsigned int n = Ta.size() - m + 1;
	unsigned int lastend=0;
	for(unsigned int i = 0; i < NUM_THREADS; ++i ){
		lastend += ceil(n / (double) NUM_THREADS);
		if(lastend > n){
			lastend = n;
		}
		int workers =  ceil(n / (double) NUM_THREADS);
		std::cout << workers<< std::endl;
		int tid = i;
		targs[tid].Ta = &Ta_d[i % nDevices];
		targs[tid].Tb = &Ta_d[i % nDevices];
		targs[tid].tid = tid;
		targs[tid].profile = Profs[tid];
		targs[tid].profileIdxs = ProfsIdxs[tid];
		targs[tid].m = m;
		targs[tid].start = i;
		targs[tid].numWorkers = ceil(n / (double) NUM_THREADS);
		//lastend = n-floor(n*sqrt(double(NUM_THREADS-i-1)/double(NUM_THREADS-i)));
		printf("val:%lf\n", sqrt(double(NUM_THREADS-i-1)/double(NUM_THREADS-i)));
		targs[tid].end = n;
		targs[tid].exclusion = m / 4;
		targs[tid].maxJoin = 0;
		printf("Launching thread %d, for start = %d, to end = %d\n", tid, targs[tid].start, targs[tid].end);
		int rc = pthread_create(&threads[tid], NULL, doThreadSTOMP, (void*) &targs[tid]);
		++tid;
	}

	for(int x = 0; x < NUM_THREADS; x++)
		pthread_join(threads[x], NULL);
	
	gpuErrchk(cudaSetDevice(0));
	thrust::device_vector<float> profile(Ta.size() - m + 1, FLT_MAX);
	thrust::device_vector<unsigned int> profileIdxs(Ta.size() - m + 1, 0);

	//Move all pieces back to the same GPU to aggregate 
	//TODO:(This can be split into steps in the case we are using a massive number of GPUs)
	for(int i = 0; i < NUM_THREADS; ++i)
	{
		if(i % nDevices != 0)
		{
			gpuErrchk(cudaSetDevice(i % nDevices));
			thrust::host_vector<unsigned long long int> temp = *Profs[i];
			delete Profs[i];
			delete ProfsIdxs[i];
			gpuErrchk(cudaSetDevice(0));
			Profs[i] = new thrust::device_vector<unsigned long long int>(temp);
			gpuErrchk( cudaPeekAtLastError() );
			ProfsIdxs[i] = new thrust::device_vector<unsigned int>(); 
			gpuErrchk( cudaPeekAtLastError() );
		}
	}
	//Compute final distance profile (Aggragate what each thread produced)
	for(int i = 0; i < NUM_THREADS; ++i){
		int curstart=0;
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(profile.begin(), profileIdxs.begin(), Profs[i] -> begin())), thrust::make_zip_iterator(thrust::make_tuple(profile.end(), profileIdxs.end(), Profs[i] -> end())), minWithIndex2());
		gpuErrchk( cudaPeekAtLastError() );
	}
	for(int i = 0; i < NUM_THREADS; ++i){
		delete Profs[i];
		delete ProfsIdxs[i];
	}
	delete [] Ta_d;
	profile_h = profile;
	profileIdxs_h = profileIdxs;
}


int main(int argc, char** argv) {


	int window_size = atoi(argv[1]);

	thrust::host_vector<DATA_TYPE> Th;
	readFile(argv[2], Th);
	//thrust::device_vector<DATA_TYPE> T;
	//T = Th;
	int size = Th.size();
	thrust::host_vector<float> profile;
	thrust::host_vector<unsigned int> profIdxs;
	printf("Starting STOMP\n");
	time_t now;
	time(&START);
	STOMP(Th,window_size,profile, profIdxs);
	time(&now);
	
	printf("Finished STOMP on %u data points in %f seconds.\n", size, difftime(now, START));
	printf("Now writing result to files\n");
	FILE* f1 = fopen( argv[3], "w");
	FILE* f2 = fopen( argv[4], "w");
	for(int i = 0; i < profIdxs.size(); ++i){
	    fprintf(f1, format_str_n, profile[i]);
	    fprintf(f2, "%u\n", profIdxs[i] + 1);
	}
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaDeviceReset());
	fclose(f1);
	fclose(f2);
    	printf("Done\n");
	return 0;
}



