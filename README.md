# STOMPSelfJoin
This is a GPU implementation of the STOMP algorithm. STOMP takes a time series as input and computes the matrix profile for a particular window size. You can read more at the [Matrix Profile Homepage](http://www.cs.ucr.edu/~eamonn/MatrixProfile.html)
# Environment
This base project requires:
 * At least version 8.0 of the CUDA toolkit available [here](https://developer.nvidia.com/cuda-toolkit).
 * An NVIDA GPU with CUDA support is also required. You can find a list of CUDA compatible GPUs [here](https://developer.nvidia.com/cuda-gpus)
 * Currently builds under linux only. 
 * We are working to create a separate repository that can generate Windows executables, it will be linked here when created. 
# Usage
* Edit the Makefile
  * Set the value of DEVICES to correspond to the number of GPU devices available on your system, this will most likely be 1 for most systems
  * Most GPU's should already be supported, but if needed set the value of ARCH to correspond to the compute capability of your GPU.
    * "-gencode=arch=compute_code,code=sm_code" where code corresponds to the compute capability or arch you wish to add.
  * Make sure CUDA_DIRECTORY corresponds to the location where cuda is installed on your system. This is usually `/usr/local/cuda-(VERSION)/` on linux
* `make`
* `STOMP window_size input_file_path output_matrix_profile_path output_indexes_path`

# Alternative Version
If you have a GPU with compute cabability 3.7, 5.2, or 6.1 you can build a more optimized version for those platforms since they have lots of shared memory
* `make best`
* `STOMPbest window_size input_file_path output_matrix_profile_path output_indexes_path`
  
# Matlab hook
If you want to use these kernels in Matlab just run `make matlab`. This will output `STOMP.ptx`. Just point Matlab to `STOMP.ptx` and `STOMP.cu` and you should be able to use any of the kernels through the [Matlab parallel computing toolbox](https://www.mathworks.com/products/parallel-computing.html).
