CC=nvcc
#Devices corresponds to the number of GPUs available on the system.
DEVICES=2
CUDA_DIRECTORY=/usr/local/cuda-7.5
ARCH=-gencode=arch=compute_61,code="sm_61,compute_61" -gencode=arch=compute_52,code="sm_52,compute_52" -gencode=arch=compute_37,code="sm_37,compute_37" -gencode=arch=compute_35,code="sm_35,compute_35" -gencode=arch=compute_50,code="sm_50,compute_50" -gencode=arch=compute_20,code="sm_20,compute_20"
ARCHBEST=-gencode=arch=compute_61,code="sm_61,compute_61" -gencode=arch=compute_52,code="sm_52,compute_52" -gencode=arch=compute_37,code="sm_37,compute_37"
LOG_FREQ=0

CNORMFLAGS=-c $(ARCH) -O3 -I$(CUDA_DIRECTORY)/include -DNUM_THREADS=$(DEVICES) -DLOG_FREQ=$(LOG_FREQ)
CBESTFLAGS=-c $(ARCHBEST) -O3 -I$(CUDA_DIRECTORY)/include -DNUM_THREADS=$(DEVICES) -DLOG_FREQ=$(LOG_FREQ) -DUSE_BEST_VERSION
MATFLAGS=-ptx -arch=$(ARCH) -O3 -I$(CUDA_DIRECTORY)/include -DNUM_THREADS=1 -DLOG_FREQ=0 
RESTARTFLAGS=-D__RESTARTING__
LDFLAGS=-L$(CUDA_DIRECTORY)/lib64 -lcufft
SOURCES=STOMP.cu
OBJECTS=STOMP.o
BESTOBJECTS=STOMPBest.o
BESTEX=STOMPbest
EXECUTABLE=STOMP

all: $(SOURCES) $(EXECUTABLE)

best: $(SOURCES) $(BESTEX)

matlab:
	nvcc $(MATFLAGS) $(SOURCES)
    
$(EXECUTABLE): $(OBJECTS)  $(SOURCES)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@
	
$(BESTEX): $(BESTOBJECTS)
	$(CC) $(LDFLAGS) $(BESTOBJECTS) -o $@


STOMP.o: $(SOURCES)
	$(CC) $(CFLAGS) STOMP.cu -o $@
	
STOMPBest.o: $(SOURCES)
	$(CC) $(CBESTFLAGS) STOMP.cu -o $@
	
clean:
	rm -f *.o STOMP STOMPrestart
