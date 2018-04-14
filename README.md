# STOMPSelfJoin
This is a GPU implementation of the STOMP algorithm. STOMP takes a time series as input and computes the matrix profile for a particular window size. You can read more at the [Matrix Profile Homepage](http://www.cs.ucr.edu/~eamonn/MatrixProfile.html)
# Environment
This base project requires:
 * At least version 9.0 of the CUDA toolkit available [here](https://developer.nvidia.com/cuda-toolkit).
 * An NVIDIA GPU with CUDA support is also required. You can find a list of CUDA compatible GPUs [here](https://developer.nvidia.com/cuda-gpus)
 * Currently builds under linux with the Makefile. 
 * Should compile under windows, but untested. 
# Usage
* Edit the Makefile
  * Most GPU's should already be supported, but if needed set the value of ARCH to correspond to the compute capability of your GPU.
    * "-gencode=arch=compute_code,code=sm_code" where code corresponds to the compute capability or arch you wish to add.
  * Make sure CUDA_DIRECTORY corresponds to the location where cuda is installed on your system. This is usually `/usr/local/cuda-(VERSION)/` on linux
* `make`
* `STOMP window_size input_file_path output_matrix_profile_path output_indexes_path (Optional: list of device numbers that you want to run on)`
* Example:
* `STOMP 1024 SampleInput/randomlist128K.txt profile.txt index.txt 0 2`
* By default, if no devices are specified, STOMP will run on all available devices


# Matlab hook (deprecated)
 * The matlab hook is deprecated. It will be left here and is still mostly functional, but it will no longer be updated.
 * If you want to use these kernels in Matlab just run `make matlab`. This will output `STOMP.ptx`.
 * Just point Matlab to `STOMP.ptx` and `STOMP.cu` and you should be able to use any of the kernels through the [Matlab parallel computing toolbox](https://www.mathworks.com/products/parallel-computing.html).
 * StompSelfJoinGPU.m is a matlab script which will compute the matrix profile using the GPU. Assuming step 1 was followed.
