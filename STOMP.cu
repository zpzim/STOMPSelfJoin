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
#include <float.h>
#include <pthread.h>
#include <math.h>



#include "cuda_profiler_api.h"

#include "STOMP.h"


#ifdef __SINGLE_PREC__
const char * format_str = "%f";
const char * format_str_n = "%f\n";
#else
const char * format_str = "%lf";
const char * format_str_n = "%lf\n";
#endif

time_t START;
struct thread_args{
	unsigned int tid;
	thrust::device_vector<DATA_TYPE> *Ta, *Tb, *profile;
	thrust::device_vector<unsigned int> *profileIdxs;
	unsigned int m;
	int exclusion;
	int maxJoin;
	int start, end;
};

struct thread_args targs[NUM_THREADS];
int nDevices;


pthread_t threads[NUM_THREADS];
//static const int NUM_THREADS = 1;
static const unsigned int WORK_SIZE = 1024;

cufftHandle plan[NUM_THREADS], plan2[NUM_THREADS], plan3[NUM_THREADS], plan4[NUM_THREADS];


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

int FileExists(const char *filename)
{
   FILE *fp = fopen (filename, "r");
   if (fp!=NULL) fclose (fp);
   return (fp!=NULL);
}

int readProgressFromFiles(int tid, thrust::device_vector<DATA_TYPE>* profile,
                            thrust::device_vector<unsigned int>*  profileIdxs,
                            thrust::device_vector<DATA_TYPE>& QTtrunc, int n, double& oldTime){
     char filenameP[20];
     char filenamePi[20];
     char filenameQT[20];
     sprintf(filenameP, "%s%d%s", "thread", tid, "p.log");
     sprintf(filenamePi, "%s%d%s", "thread", tid, "i.log");
     sprintf(filenameQT, "%s%d%s", "thread", tid, "QT.log");
     FILE* fP = fopen( filenameP, "r");
     FILE* fi = fopen( filenamePi, "r");
     FILE* fQT = fopen( filenameQT, "r");
     thrust::host_vector<DATA_TYPE> prof(n);
     thrust::host_vector<unsigned int> idxs(n);
     thrust::host_vector<DATA_TYPE> QT(n);
     DATA_TYPE num;
     unsigned int idx;
     for(int i = 0; i < prof.size(); ++i){
        fscanf(fP, format_str, &num);
        prof[i] = num;
        fscanf(fi, "%u", &idx);
        idxs[i] = idx;
     }
     int start;
     fscanf(fQT, "%d", &start);
     fscanf(fQT, "%lf", &oldTime);
     printf("Read files, going to restart from start = %d\n", start);
     for(int i = 0; i < QT.size(); ++i){
        
        fscanf(fQT, format_str, &num);
        QT[i] = num;
     } 
     fclose(fP);
     fclose(fi);
     fclose(fQT);
     *profile = prof;
     *profileIdxs = idxs;
     QTtrunc = QT;            
     return start;        
} 



void writeProgressToDisk(int tid, thrust::device_vector<DATA_TYPE>* profile,
                            thrust::device_vector<unsigned int>*  profileIdxs,
                            thrust::device_vector<DATA_TYPE>& QTtrunc, int curr, double timetaken ){
     printf("writing progress to files curr = %d\n", curr);
     char filenameP[20];
     char filenamePOld[20];
     char filenamePi[20];
     char filenamePiOld[20];
     char filenameQT[20];
     char filenameQTOld[20];
     
     sprintf(filenameP, "%s%d%s", "thread", tid, "p.log");
     sprintf(filenamePi, "%s%d%s", "thread", tid, "i.log");
     sprintf(filenameQT, "%s%d%s", "thread", tid, "QT.log");
     sprintf(filenamePOld, "%s%d%s", "thread", tid, "p.old");
     sprintf(filenamePiOld, "%s%d%s", "thread", tid, "i.old");
     sprintf(filenameQTOld, "%s%d%s", "thread", tid, "QT.old");
     
     //If file already exists move it, we don't want to have a period of time where
     //if the program fails we lose our data.
     if(FileExists(filenameP)){
         rename(filenameP, filenamePOld);
     }if(FileExists(filenamePi)){
         rename(filenamePi, filenamePiOld);
     }if(FileExists(filenameQT)){
         rename(filenameQT, filenameQTOld);
     }
     FILE* fP = fopen( filenameP, "w");
     FILE* fi = fopen( filenamePi, "w");
     FILE* fQT = fopen( filenameQT, "w");
     thrust::host_vector<DATA_TYPE> prof = *profile;
     thrust::host_vector<unsigned int> idxs = *profileIdxs;
     thrust::host_vector<DATA_TYPE> QT = QTtrunc;
     for(int i = 0; i < prof.size(); ++i){
        fprintf(fP, format_str_n, prof[i]);  
     }
     for(int i = 0; i < idxs.size(); ++i){
        fprintf(fi, "%u\n", idxs[i]);
     }
     //write the iteration in 
     fprintf(fQT, "%d\n", curr);
     fprintf(fQT, "%lf\n", timetaken);
     for(int i = 0; i < QTtrunc.size(); ++i){      
        fprintf(fQT, format_str_n, QT[i]);
     } 
     fclose(fP);
     fclose(fi);
     fclose(fQT);  
     printf("done writing\n");    
}

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
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
		D[a] = _MAX_VAL_;
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

// Reduction kernel, upper layer
template <unsigned int blockSize>      
__global__ void reduce(const DATA_TYPE *g_idata, DATA_TYPE *g_odata, unsigned int *g_oloc,  unsigned int n) {
	__shared__ DATA_TYPE sdata[blockSize];
	__shared__ DATA_TYPE sloc[blockSize];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	DATA_TYPE temp;
	unsigned int temploc;
	sdata[tid] = _MAX_VAL_;
	while (i < n) {
		if (i + blockSize <n)
		{
			if (g_idata[i] < g_idata[i+blockSize])
			{
				temp=g_idata[i];
				temploc=i;
			}
			else
			{
				temp=g_idata[i+blockSize];
				temploc = i+blockSize;
			}
		}
		else
		{
			temp = g_idata[i];
			temploc = i;
		}
		if (sdata[tid] > temp)
		{
			sdata[tid] = temp;
			sloc[tid] = temploc;
		}
		i += gridSize; 
	}
	__syncthreads();
	if (blockSize >= 1024) { 
		if (tid < 512 && sdata[tid] > sdata[tid + 512])
		{
			sdata[tid] = sdata[tid + 512];
			sloc[tid] = sloc[tid + 512];
		}
		__syncthreads();
	}
	if (blockSize >= 512 ) { 	
		if (tid < 256 && sdata[tid] > sdata[tid + 256]) 
		{
			sdata[tid] = sdata[tid + 256];
			sloc[tid] = sloc[tid + 256];
		}
		__syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128 && sdata[tid] > sdata[tid + 128])
		{
			sdata[tid] = sdata[tid + 128]; 
			sloc[tid] = sloc[tid + 128];
		}
		__syncthreads(); 
	}
	if (blockSize >= 128) { 
		if (tid < 64 && sdata[tid] > sdata[tid + 64])  
		{
			sdata[tid] = sdata[tid + 64]; 
			sloc[tid] = sloc[tid + 64];
		}
		__syncthreads(); 
	}
	
	if (blockSize >= 64) {
		if (tid < 32 && sdata[tid] > sdata[tid + 32])  
		{
			sdata[tid] = sdata[tid + 32]; 
			sloc[tid] = sloc[tid + 32];
		}
		__syncthreads(); 
	}

	if (blockSize >= 32) {
		if (tid < 16 && sdata[tid] > sdata[tid + 16])  
		{
			sdata[tid] = sdata[tid + 16]; 
			sloc[tid] = sloc[tid + 16];
		}
		__syncthreads(); 
	}

	if (blockSize >= 16) {
		if (tid < 8 && sdata[tid] > sdata[tid + 8])  
		{
			sdata[tid] = sdata[tid + 8]; 
			sloc[tid] = sloc[tid + 8];
		}
		__syncthreads(); 
	}

	if (blockSize >= 8) {
		if (tid < 4 && sdata[tid] > sdata[tid + 4])  
		{
			sdata[tid] = sdata[tid + 4]; 
			sloc[tid] = sloc[tid + 4];
		}
		__syncthreads(); 
	}

	if (blockSize >= 4) {
		if (tid < 2 && sdata[tid] > sdata[tid + 2])  
		{
			sdata[tid] = sdata[tid + 2]; 
			sloc[tid] = sloc[tid + 2];
		}
		__syncthreads(); 
	}

	if (blockSize >= 2) {
		if (tid == 0) 
		{
			if (sdata[0] <= sdata[1])
			{
				g_odata[blockIdx.x] = sdata[0];
				g_oloc[blockIdx.x] = sloc[0];
			}
			else
			{
				g_odata[blockIdx.x] = sdata[1];
				g_oloc[blockIdx.x] = sloc[1];
			}
		}
	}
	else
	{
		if (tid == 0) 
		{
			g_odata[blockIdx.x] = sdata[0];
			g_oloc[blockIdx.x] = sloc[0];
		}
	}
}

//reduction kernel, lower layer
template <unsigned int blockSize>      
__global__ void reducelast(DATA_TYPE *g_idata, unsigned int *g_iloc, unsigned int start_loc, DATA_TYPE* profilei, unsigned int* profileidxi, unsigned int n) {

	__shared__ DATA_TYPE sdata[blockSize];
	__shared__ DATA_TYPE sloc[blockSize];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	DATA_TYPE temp;
	unsigned int temploc;
	sdata[tid] = _MAX_VAL_;
	DATA_TYPE minval;
	unsigned int minloc;
	while (i < n) {
		if (i + blockSize <n)
		{
			if (g_idata[i] < g_idata[i+blockSize])
			{
				temp=g_idata[i];
				temploc=g_iloc[i];
			}
			else
			{
				temp=g_idata[i+blockSize];
				temploc = g_iloc[i+blockSize];
			}
		}
		else
		{
			temp = g_idata[i];
			temploc = g_iloc[i];
		}
		if (sdata[tid] > temp)
		{
			sdata[tid] = temp;
			sloc[tid] = temploc;
		}
		i += gridSize; 
	}
	__syncthreads();
	if (blockSize >= 1024) { 
		if (tid < 512 && sdata[tid] > sdata[tid + 512])
		{
			sdata[tid] = sdata[tid + 512];
			sloc[tid] = sloc[tid + 512];
		}
		__syncthreads();
	}
	if (blockSize >= 512 ) { 	
		if (tid < 256 && sdata[tid] > sdata[tid + 256]) 
		{
			sdata[tid] = sdata[tid + 256];
			sloc[tid] = sloc[tid + 256];
		}
		__syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128 && sdata[tid] > sdata[tid + 128])
		{
			sdata[tid] = sdata[tid + 128]; 
			sloc[tid] = sloc[tid + 128];
		}
		__syncthreads(); 
	}
	if (blockSize >= 128) { 
		if (tid < 64 && sdata[tid] > sdata[tid + 64])  
		{
			sdata[tid] = sdata[tid + 64]; 
			sloc[tid] = sloc[tid + 64];
		}
		__syncthreads(); 
	}
	
	if (blockSize >= 64) {
		if (tid < 32 && sdata[tid] > sdata[tid + 32])  
		{
			sdata[tid] = sdata[tid + 32]; 
			sloc[tid] = sloc[tid + 32];
		}
		__syncthreads(); 
	}

	if (blockSize >= 32) {
		if (tid < 16 && sdata[tid] > sdata[tid + 16])  
		{
			sdata[tid] = sdata[tid + 16]; 
			sloc[tid] = sloc[tid + 16];
		}
		__syncthreads(); 
	}

	if (blockSize >= 16) {
		if (tid < 8 && sdata[tid] > sdata[tid + 8])  
		{
			sdata[tid] = sdata[tid + 8]; 
			sloc[tid] = sloc[tid + 8];
		}
		__syncthreads(); 
	}

	if (blockSize >= 8) {
		if (tid < 4 && sdata[tid] > sdata[tid + 4])  
		{
			sdata[tid] = sdata[tid + 4]; 
			sloc[tid] = sloc[tid + 4];
		}
		__syncthreads(); 
	}

	if (blockSize >= 4) {
		if (tid < 2 && sdata[tid] > sdata[tid + 2])  
		{
			sdata[tid] = sdata[tid + 2]; 
			sloc[tid] = sloc[tid + 2];
		}
		__syncthreads(); 
	}

	if (blockSize >= 2) {
		if (tid == 0) 
		{
			if (sdata[0] <= sdata[1])
			{
				minval = sdata[0];
				minloc = sloc[0];
			}
			else
			{
				minval = sdata[1];
				minloc = sloc[1];
			}
		}
	}
	else
	{
		if (tid == 0) 
		{
			minval = sdata[0];
			minloc = sloc[0];
		}
	}

	if (tid==0)
	{
		if (minval<(*profilei))
		{
			//printf("Here\n");
			(*profilei)=minval;
			(*profileidxi)=minloc+start_loc;
		}
	}

}



//The main update function, this updates QT, the dist profile, and the matrix profile
__global__ void Update(DATA_TYPE* QT, DATA_TYPE* QTtemp, DATA_TYPE* QTb, DATA_TYPE* Ta, DATA_TYPE* Tb, DATA_TYPE* D, DATA_TYPE* Means, DATA_TYPE* stds, DATA_TYPE* profile, unsigned int* profileIdxs, int i, unsigned int m, unsigned int sz){
    	const DATA_TYPE Qmean = Means[i];
	const DATA_TYPE Qstd = stds[i];
    	const DATA_TYPE x = Tb[i - 1];
	const DATA_TYPE y = Tb[i + m - 1];
	const int exclusionZone = m / 4; 
	int a = blockIdx.x * blockDim.x + threadIdx.x + i;
	if(a < sz){
		QT[a] = QTtemp[a - 1] - Ta[a - 1] * x + Ta[a + m - 1] * y;
		if (a > i - exclusionZone && a < i + exclusionZone ){
			D[a] = _MAX_VAL_;
		}
		else{
			D[a] = sqrt(abs(2 * (m - (QT[a] - m * Means[a] * Qmean) / (stds[a] * Qstd))));
		}
		if(D[a] < profile[a]){
	    		profile[a] = D[a];
	    		profileIdxs[a] = i;
		}
	}
}

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
	thrust::device_vector<DATA_TYPE> Ta(T.size() * 2);
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

void reducemain(thrust::device_vector<DATA_TYPE>& vd, unsigned int start_loc, unsigned int max_block_num, unsigned int max_thread_num, unsigned int n, thrust::device_vector<DATA_TYPE>* profile, thrust::device_vector<unsigned int>* profileidx, unsigned int i, thrust::device_vector<DATA_TYPE>& reduced_result, thrust::device_vector<unsigned int>& reduced_loc)
{

	if (n==0) //if this happens, there's an error
		return;
	if (max_thread_num>1024)
		max_thread_num=1024;
	
	unsigned int * middle_loc_pointer=reduced_loc.data().get();

	
	unsigned int num_threads=max_thread_num;
	
	unsigned int num_blocks=n/(num_threads*2);
	if (n%(num_threads*2)!=0)
		num_blocks++;
	if (num_blocks>=max_block_num)
		num_blocks=max_block_num;
	DATA_TYPE * middle_pointer=NULL;
	unsigned int curn;
	if (num_blocks>1) //upperlevel reduction
	{
		middle_pointer=reduced_result.data().get();
		curn=num_blocks;
		switch (num_threads)
		{
			case 1024:
				reduce<1024><<<num_blocks,1024>>>(vd.data().get()+start_loc,reduced_result.data().get(),reduced_loc.data().get(),n); break;
			case 512:
				reduce<512><<<num_blocks,512>>>(vd.data().get()+start_loc,reduced_result.data().get(),reduced_loc.data().get(),n); break;
			case 256:
				reduce<256><<<num_blocks,256>>>(vd.data().get()+start_loc,reduced_result.data().get(),reduced_loc.data().get(),n); break;
			case 128:
				reduce<128><<<num_blocks,128>>>(vd.data().get()+start_loc,reduced_result.data().get(),reduced_loc.data().get(),n); break;
			case 64:
				reduce<64><<<num_blocks,64>>>(vd.data().get()+start_loc,reduced_result.data().get(),reduced_loc.data().get(),n); break;
			case 32:
				reduce<32><<<num_blocks,32>>>(vd.data().get()+start_loc,reduced_result.data().get(),reduced_loc.data().get(),n); break;
			case 16:
				reduce<16><<<num_blocks,16>>>(vd.data().get()+start_loc,reduced_result.data().get(),reduced_loc.data().get(),n); break;
			case 8:
				reduce<8><<<num_blocks,8>>>(vd.data().get()+start_loc,reduced_result.data().get(),reduced_loc.data().get(),n); break;
			case 4:
				reduce<4><<<num_blocks,4>>>(vd.data().get()+start_loc,reduced_result.data().get(),reduced_loc.data().get(),n); break;
			case 2:
				reduce<2><<<num_blocks,2>>>(vd.data().get()+start_loc,reduced_result.data().get(),reduced_loc.data().get(),n); break;
			default:
				break;
		}
	        gpuErrchk( cudaPeekAtLastError() );
	}
	else
	{
		middle_pointer=vd.data().get()+start_loc;
		curn=n;
		thrust::sequence(reduced_loc.begin(),reduced_loc.begin()+curn);
	}


	//printf("curn:%u\n",curn);
	num_threads=floor(pow(2,ceil(log(curn)/log(2))-1));
	//printf("num_threads:%u\n",num_threads);
	if (num_threads>max_thread_num)
		num_threads=max_thread_num;
	//printf("Num of threads:%u\nCurn:%u\nNum of blocks:%u\n", num_threads, curn, num_blocks);
	switch (num_threads)
	{
		case 1024:
			reducelast<1024><<<1,1024>>>(middle_pointer, middle_loc_pointer, start_loc, (*profile).data().get()+i, (*profileidx).data().get()+i, curn);
	        	gpuErrchk( cudaPeekAtLastError() );
			break;
		case 512:
			reducelast<512><<<1,512>>>(middle_pointer,middle_loc_pointer, start_loc, (*profile).data().get()+i, (*profileidx).data().get()+i,  curn); 
	        	gpuErrchk( cudaPeekAtLastError() );
			break;
		case 256:
			reducelast<256><<<1,256>>>(middle_pointer,middle_loc_pointer, start_loc, (*profile).data().get()+i, (*profileidx).data().get()+i, curn);
	        	gpuErrchk( cudaPeekAtLastError() );
			break;
		case 128:
			reducelast<128><<<1,128>>>(middle_pointer,middle_loc_pointer, start_loc,  (*profile).data().get()+i, (*profileidx).data().get()+i,curn); 
	        	gpuErrchk( cudaPeekAtLastError() );
			break;
		case 64:
			reducelast<64><<<1,64>>>(middle_pointer,middle_loc_pointer, start_loc,  (*profile).data().get()+i, (*profileidx).data().get()+i,curn);
	        	gpuErrchk( cudaPeekAtLastError() );
			break;
		case 32:
			reducelast<32><<<1,32>>>(middle_pointer,middle_loc_pointer, start_loc, (*profile).data().get()+i, (*profileidx).data().get()+i, curn); 
	        	gpuErrchk( cudaPeekAtLastError() );
			break;
		case 16:
			reducelast<16><<<1,16>>>(middle_pointer,middle_loc_pointer, start_loc, (*profile).data().get()+i, (*profileidx).data().get()+i,curn); 
	        	gpuErrchk( cudaPeekAtLastError() );
			break;
		case 8:
			reducelast<8><<<1,8>>>(middle_pointer,middle_loc_pointer, start_loc, (*profile).data().get()+i, (*profileidx).data().get()+i,curn); 
	        	gpuErrchk( cudaPeekAtLastError() );
			break;
		case 4:
			reducelast<4><<<1,4>>>(middle_pointer,middle_loc_pointer, start_loc,  (*profile).data().get()+i, (*profileidx).data().get()+i,curn); 
	        	gpuErrchk( cudaPeekAtLastError() );
			break;
		case 2:
			reducelast<2><<<1,2>>>(middle_pointer,middle_loc_pointer, start_loc, (*profile).data().get()+i, (*profileidx).data().get()+i, curn); 
	        	gpuErrchk( cudaPeekAtLastError() );
			break;
		case 1:
			reducelast<1><<<1,1>>>(middle_pointer,middle_loc_pointer, start_loc, (*profile).data().get()+i, (*profileidx).data().get()+i, curn); 
	        	gpuErrchk( cudaPeekAtLastError() );
			break;
		case 0:
			reducelast<1><<<1,1>>>(middle_pointer,middle_loc_pointer, start_loc, (*profile).data().get()+i, (*profileidx).data().get()+i, curn); 
	        	gpuErrchk( cudaPeekAtLastError() );
			break;
		default:
			break;
	}	
}


//Performs STOMP algorithm
void* doThreadSTOMP(void* argsp){
    
    	thread_args* args = (thread_args*) argsp;
    	int tid = args->tid;
    	gpuErrchk(cudaSetDevice(tid % nDevices));
	thrust::device_vector<DATA_TYPE>* Ta = args -> Ta;
 	thrust::device_vector<DATA_TYPE>* Tb = args -> Tb;
	thrust::device_vector<DATA_TYPE>* profile = args -> profile;
	thrust::device_vector<unsigned int>* profileIdxs = args -> profileIdxs;
	
	int m = args -> m;
	int exclusion = args -> exclusion;
	int maxJoin = args -> maxJoin;
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

	//variables for reduction
	const unsigned int max_block_num=2048;
	const unsigned int max_thread_num=1024;

	thrust::device_vector<DATA_TYPE> reduced_result(max_block_num);
	unsigned int middle_loc_size=max_block_num>max_thread_num?max_block_num:max_thread_num;
	thrust::device_vector<unsigned int> reduced_loc(middle_loc_size);

#ifndef __RESTARTING__

    	printf("allocating Qa\n");
	thrust::device_vector<DATA_TYPE> Qa(m), Qb(m);
	gpuErrchk( cudaPeekAtLastError() );

    	thrust::copy(thrust::cuda::par,Tb -> begin() + start, Tb -> begin() + start + m, Qb.begin());
    	gpuErrchk( cudaPeekAtLastError() );
	thrust::copy(thrust::cuda::par,Ta -> begin(), Ta -> begin() + m, Qa.begin());
	gpuErrchk( cudaPeekAtLastError() );
	printf("allocating QT\n");
	thrust::device_vector<DATA_TYPE> QT;
	
	
	SlidingDotProducts(Qb, *Ta, QT, plan[tid], plan2[tid]);
    	printf("allocating QTTrunc\n");
	thrust::device_vector<DATA_TYPE> QTtrunc(n);
	gpuErrchk( cudaPeekAtLastError() );

	thrust::copy(thrust::cuda::par,QT.begin() + m - 1, QT.begin() + m + n - 1, QTtrunc.begin());
	gpuErrchk( cudaPeekAtLastError() );
	//thrust::copy(thrust::cuda::par,QTb.begin() + m - 1, QTb.begin() + m + n2 - 1, QTbtrunc.begin()); 
	QT.clear();
	QT.shrink_to_fit();
	
	SlidingDotProducts(Qa, *Tb, QT, plan[tid], plan2[tid]);
	printf("allocating QTbtrunc\n");
	thrust::device_vector<DATA_TYPE> QTbtrunc(n);
	thrust::copy(thrust::cuda::par,QT.begin() + m - 1, QT.begin() + m + n - 1, QTbtrunc.begin());
	
	QT.clear();
	QT.shrink_to_fit();
    
	
	printf("allocating Means/stds\n");
	thrust::device_vector<DATA_TYPE> Means(n), stds(n), squares(Ta -> size()), sums(Ta -> size());
	printf("allocating Means/stdsb\n");
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
	
    CalculateDistProfile<<<grid, block>>>(QTtrunc.data().get(), D.data().get(), Means.data().get(), stds.data().get(), m, start, n);
	gpuErrchk( cudaPeekAtLastError() );


    *profile = D;
    	
    //Initialize the indexes to the starting position
	profileIdxs -> resize(n, start);
	if (n>1)
		reducemain(D, start+1, 2048, 1024, n-1-start, profile, profileIdxs, start, reduced_result, reduced_loc);
    
#else

    //We are restarting, so allocate required variables and load data from files
    thrust::device_vector<DATA_TYPE> QT;
    thrust::device_vector<DATA_TYPE> Qa(m);
    thrust::copy(thrust::cuda::par,Ta -> begin(), Ta -> begin() + m, Qa.begin());
    
    //Generate the initial value of QT at i = 0
    SlidingDotProducts(Qa, *Tb, QT, plan[tid], plan2[tid]);
	printf("allocating QTbtrunc\n");
	thrust::device_vector<DATA_TYPE> QTbtrunc(n);
	thrust::copy(thrust::cuda::par,QT.begin() + m - 1, QT.begin() + m + n - 1, QTbtrunc.begin());
	QT.clear();
	QT.shrink_to_fit();
    thrust::device_vector<DATA_TYPE> QTtrunc(n);
    start = readProgressFromFiles(tid, profile, profileIdxs, QTtrunc, n, oldTime);
	printf("allocating Means/stds\n");
	thrust::device_vector<DATA_TYPE> Means(n), stds(n), squares(Ta -> size()), sums(Ta -> size());
	printf("allocating Means/stdsb\n");
	thrust::inclusive_scan(Ta -> begin(),Ta -> end(),sums.begin(), op1);
	thrust::transform_inclusive_scan(Ta -> begin(), Ta -> end(), squares.begin(), op2,op1);
	slidingMean<<<grid, block>>>(sums.data().get(),m, n, Means.data().get());
	slidingStd<<<grid, block>>>(squares.data().get(), m, n, Means.data().get(), stds.data().get());
	sums.clear();
	squares.clear();
	sums.shrink_to_fit();
	squares.shrink_to_fit();
	
	thrust::device_vector<DATA_TYPE> D;
	D.resize(n,_MAX_VAL_);
    
#endif    
    
    printf("Allocating QTtemp\n");
    thrust::device_vector<DATA_TYPE> QTtrunc2(n);
    printf("Copying QTtrunc\n");

    time_t start2, now2;
    time_t lastLogged;
    time(&start2);
    time(&lastLogged);
    bool usingSecondary = true;
    bool fileOne = true;
   // cudaProfilerInitialize();
	for(unsigned int i = start + 1; i < end; ++i){// end; ++i){
	    dim3 grid3_cur((n-i) / WORK_SIZE + 1, 1, 1);
	    //Swap buffers every iteration so we don't have to copy anything explicitly.
	    if(usingSecondary){
	        //cudaProfilerStart();

	        Update<<<grid3_cur,block>>>(QTtrunc2.data().get(), QTtrunc.data().get(), QTbtrunc.data().get(), Ta -> data().get(), Tb -> data().get(), D.data().get(), Means.data().get(), stds.data().get(), profile -> data().get(), profileIdxs -> data().get(), i,  m, n);
	       // cudaProfilerStop();
	        gpuErrchk( cudaPeekAtLastError() );
		//if((i - start) % 10000 == 1){
		//	time(&now2);
                 //       printf("Spent %f seconds over the last 10000 iterations\n",difftime(now2, start2));
               //         writeProgressToDisk(tid, profile, profileIdxs, QTtrunc2, i);
                 //       time(&start2);

		//}
		if (i<n-1)
		{
			reducemain(D, i+1, 2048, 1024, n-i-1, profile, profileIdxs, i, reduced_result, reduced_loc);
		}
		 time(&now2);
		 if((int) difftime(now2, start2) % 60 == 0 && difftime(now2,lastLogged) > 2){
				printf("Thread %d finished iteration %u, %f percent iterations done: current total time taken = %lf seconds\n", tid, i, (i - start)/((float)(end - start)), difftime(now2, START) + oldTime);
				time(&lastLogged);
		}
	        if(LOG_FREQ != 0 && difftime(now2, start2) > LOG_FREQ){
		        
		        //printf("Thread %d Spent %lf seconds since last save\n", tid, difftime(now2, start2));
		        writeProgressToDisk(tid, profile, profileIdxs, QTtrunc2, i, difftime(now2, START) + oldTime);
		        time(&start2);
		    }
		    usingSecondary = false;
	    }else{
	        Update<<<grid3_cur,block>>>(QTtrunc.data().get(), QTtrunc2.data().get(), QTbtrunc.data().get(), Ta -> data().get(), Tb -> data().get(), D.data().get(), Means.data().get(), stds.data().get(), profile -> data().get(), profileIdxs -> data().get(), i, m, n);
	        gpuErrchk( cudaPeekAtLastError() );

		if (i<n-1)
		{
			reducemain(D, i+1, 2048, 1024, n-i-1, profile, profileIdxs, i, reduced_result, reduced_loc);
		}
		time(&now2);

                if((int) difftime(now2, start2) % 60 == 0 && difftime(now2,lastLogged) > 2){
				printf("Thread %d finished iteration %u, %f percent iterations done: current total time taken = %lf seconds\n", tid, i, (i - start)/((float)(end - start)), difftime(now2, START) + oldTime);
				time(&lastLogged);
		}
	        if(LOG_FREQ != 0 && difftime(now2, start2) > LOG_FREQ){
		        //time(&now2);
		        //printf("Thread %d Spent %lf seconds since last save\n",tid, difftime(now2, start2));
		        writeProgressToDisk(tid, profile, profileIdxs, QTtrunc, i, difftime(now2, START) + oldTime);
		        time(&start2);
		    }
		    usingSecondary = true;
	    }
		
	}
	time_t now3;
	time(&now3);
	printf("Finished thread %d over all iterations in %lf seconds\n", tid, difftime(now3, START) + oldTime);
	pthread_exit(0);
}


//Allocates threads on a CPU to distribute work to each specified device
__host__ void STOMP(thrust::host_vector<DATA_TYPE>& Ta, unsigned int m,
		    thrust::host_vector<DATA_TYPE>& profile_h, thrust::host_vector<unsigned int>& profileIdxs_h){
	
	gpuErrchk(cudaGetDeviceCount(&nDevices));
	STOMPinit(Ta.size());
	thrust::device_vector<DATA_TYPE>* Ta_d = new thrust::device_vector<DATA_TYPE>[nDevices];
	thrust::device_vector<DATA_TYPE>* Profs[NUM_THREADS];
	thrust::device_vector<unsigned int>* ProfsIdxs[NUM_THREADS];
	printf("HERE\n");
	for(int i = 0; i < nDevices; ++i){
	    gpuErrchk(cudaSetDevice(i));
	    Ta_d[i] = Ta; 
	}
	printf("HERE2\n");
	for(int i = 0; i < NUM_THREADS; ++i){
	    gpuErrchk(cudaSetDevice(i % nDevices));
	    Profs[i] = new thrust::device_vector<DATA_TYPE>();
	    ProfsIdxs[i] = new thrust::device_vector<unsigned int>();
	}
	printf("HERE3\n");
	gpuErrchk(cudaSetDevice(0));
	unsigned int n = Ta.size() - m + 1;
	//unsigned int t_work_size = ceil(n / (double) NUM_THREADS);
	unsigned int lastend=0;
	for(unsigned int i = 0; i < NUM_THREADS; ++i ){
		int tid = i;
		targs[tid].Ta = &Ta_d[i % nDevices];
		targs[tid].Tb = &Ta_d[i % nDevices];
		targs[tid].tid = tid;
		targs[tid].profile = Profs[tid];
		targs[tid].profileIdxs = ProfsIdxs[tid];
		targs[tid].m = m;
		targs[tid].start = lastend;
		lastend = n-floor(n*sqrt(double(NUM_THREADS-i-1)/double(NUM_THREADS-i)));
		printf("val:%lf\n", sqrt(double(NUM_THREADS-i-1)/double(NUM_THREADS-i)));
		targs[tid].end = lastend;
		targs[tid].exclusion = m / 4;
		targs[tid].maxJoin = 0;
		printf("Launching thread %d, for start = %d, to end = %d\n", tid, targs[tid].start, targs[tid].end);
		int rc = pthread_create(&threads[tid], NULL, doThreadSTOMP, (void*) &targs[tid]);
		++tid;
	}

	for(int x = 0; x < NUM_THREADS; x++)
		pthread_join(threads[x], NULL);
	
	gpuErrchk(cudaSetDevice(0));
    thrust::device_vector<DATA_TYPE> profile(Ta.size() - m + 1, _MAX_VAL_);
    thrust::device_vector<unsigned int> profileIdxs(Ta.size() - m + 1, 0);
    // = profile.resize(Ta.size() - m + 1, _MAX_VAL_);
	//profileIdxs.resize(Ta.size() - m + 1, 0);
	
	for(int i = 0; i < NUM_THREADS; ++i){
	    if(i % nDevices != 0){
	        gpuErrchk(cudaSetDevice(i % nDevices));
	        thrust::host_vector<DATA_TYPE> temp = *Profs[i];
	        thrust::host_vector<DATA_TYPE> temp2 = *ProfsIdxs[i];
	        delete Profs[i];
	        delete ProfsIdxs[i];
	        gpuErrchk(cudaSetDevice(0));
	        Profs[i] = new thrust::device_vector<DATA_TYPE>(temp);
	        gpuErrchk( cudaPeekAtLastError() );
	        ProfsIdxs[i] = new thrust::device_vector<unsigned int>(temp2); 
	        gpuErrchk( cudaPeekAtLastError() );
	    }
	    
	
	}
	//Compute final distance profile (Aggragate what each thread produced)
	for(int i = 0; i < NUM_THREADS; ++i){
		int curstart=targs[i].start;
			thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(profile.begin() + curstart, Profs[i] -> begin() + curstart, profileIdxs.begin() + curstart, ProfsIdxs[i]->begin() + curstart)), thrust::make_zip_iterator(thrust::make_tuple(profile.end(), Profs[i] -> end(), profileIdxs.end(), ProfsIdxs[i] -> end())), minWithIndex2());
			 gpuErrchk( cudaPeekAtLastError() );
	}
    for(int i = 0; i < NUM_THREADS; ++i){
        printf("HELLO\n");
        delete Profs[i];
        delete ProfsIdxs[i];
        printf("HELLO2\n");
        
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
	thrust::host_vector<DATA_TYPE> profile;
	thrust::host_vector<unsigned int> profIdxs;
	printf("Starting STOMP\n");
	time_t now;
	time(&START);
	STOMP(Th,window_size,profile, profIdxs);
	time(&now);
	
	printf("Finished STOMP on %u data points in %f seconds.\n", size, difftime(now, START) + );
	printf("Now writing result to files\n");
	//thrust::host_vector<DATA_TYPE> p = profile;
	//printf("Copied profile back to host\n");
	//thrust::host_vector<unsigned int> pi = profIdxs;
	FILE* f1 = fopen( argv[3], "w");
	//fprintf(f1, "Finished STOMP on %u data points in %f seconds.\n", size, difftime(now, START));
	FILE* f2 = fopen( argv[4], "w");
	for(int i = 0; i < profIdxs.size(); ++i){
	    fprintf(f1, format_str_n, profile[i]);
	    fprintf(f2, "%u\n", profIdxs[i] + 1);
		//printf("%f, %u\n", p[i], pi[i] + 1);
		//printf("%u\n",pi[i] + 1);
		//printf("%f\n",p[i]);
	}
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaDeviceReset());
	fclose(f1);
	fclose(f2);
    printf("Done\n");
	return 0;
}


