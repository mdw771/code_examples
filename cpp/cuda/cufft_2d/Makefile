CC = nvcc
CFLAGS = --ptxas-options=-v --use_fast_math -l cufft -L /usr/local/cuda-10.1/lib64
#LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64

all:fft.cu
	$(RM) fft
	$(CC) $(CFLAGS) -o fft fft.cu 

clean:
	$(RM) fft
