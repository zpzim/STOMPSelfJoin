CUDA_DIRECTORY=/usr/local/cuda-8.0
CC=$(CUDA_DIRECTORY)/bin/nvcc
ARCH=-gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_37,code=sm_37
CFLAGS=-c -std=c++11 $(ARCH) -O3 -I$(CUDA_DIRECTORY)/include

#If you want to use the matlab hook, please specify the compute capability of your GPU below
MATLABARCH=-gencode=arch=compute_37,code=sm_37 
MATFLAGS=-ptx -std=c++11 $(MATLABARCH) -O3 -I$(CUDA_DIRECTORY)/include 
LDFLAGS=-L$(CUDA_DIRECTORY)/lib64 -lcufft
SOURCES=STOMP.cu
HEADERS=STOMP.h
OBJECTS=STOMP.o
EXECUTABLE=STOMP

all: $(SOURCES) $(EXECUTABLE)

best: $(SOURCES) $(BESTEX)

matlab: $(SOURCES) $(HEADERS)
	$(CC) $(MATFLAGS) $(SOURCES)
    
$(EXECUTABLE): $(OBJECTS)  $(SOURCES) $(HEADERS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@
	
STOMP.o: $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) STOMP.cu -o $@
	
clean:
	rm -f *.o STOMP
