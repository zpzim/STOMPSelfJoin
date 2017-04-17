# STOMPSelfJoin
This is a GPU implementation of the STOMP algorithm. STOMP takes a time series as input and computes the matrix profile for a particular window size. You can read more at the [Matrix Profile Homepage](http://www.cs.ucr.edu/~eamonn/MatrixProfile.html)
# Usage
* Edit the Makefile
  * Set the value of DEVICES to correspond to the number of GPU devices available on your system, this will most likely be 1 for most systems
  * Set the value of ARCH to correspond to the compute capability of your GPU
  * Make sure CUDA_DIRECTORY corresponds to the location where cuda is installed on your system. This is usually /usr/local/cuda-(VERSION)/ on linux
* `make`
* `STOMP window_size input_file_path output_matrix_profile_path output_indexes_path`
  

