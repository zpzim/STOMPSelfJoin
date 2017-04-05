CC=nvcc
#Devices corresponds to the number of GPUs available on the system.
DEVICES=2
#Set ARCH to the cuda architecture corresponding to your GPU
ARCH=sm_37
LOG_FREQ=0
CUDA_DIRECTORY=/usr/local/cuda-7.5
CFLAGS=-c -arch=$(ARCH) -O3 -I$(CUDA_DIRECTORY)/include -DNUM_THREADS=$(DEVICES) -DLOG_FREQ=$(LOG_FREQ) #-Xptxas -dlcm=cg #-D__SINGLE_PREC__
RESTARTFLAGS=-D__RESTARTING__
LDFLAGS=-L$(CUDA_DIRECTORY)/lib64 -lcufft
SOURCES=STOMP.cu
OBJECTS=STOMP.o
RESTARTEX=STOMPrestart
RESTARTOBJECTS=STOMPrestart.o
EXECUTABLE=STOMP

all: $(SOURCES) $(EXECUTABLE)

restart: $(SOURCES) $(RESTARTEX)


    
$(EXECUTABLE): $(OBJECTS)  $(SOURCES)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@
	
$(RESTARTEX): $(RESTARTOBJECTS)
	$(CC) $(LDFLAGS) $(RESTARTOBJECTS) -o $@

STOMPrestart.o:
	$(CC) $(CFLAGS) $(RESTARTFLAGS) STOMP.cu -o $@

STOMP.o: $(SOURCES)
	$(CC) $(CFLAGS) STOMP.cu -o $@
	
clean:
	rm -f *.o STOMP STOMPrestart
