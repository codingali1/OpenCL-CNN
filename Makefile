CC 	= gcc
CFLAGS	= -O2
OBJECT	= OpenCL_CNN.c cnn_opencl.c Compare_result.c
TARGET	= cnn_opencl

LIB_NAMES = -lOpenCL -lm
DIR_LIB_INTEL_OPENCL = -L/opt/intel/system_studio_2020/opencl/SDK/lib64/
DIR_HEADER_INTEL_OPENCL = -I/opt/intel/system_studio_2020/opencl/SDK/include/

DIR_LIB_NVIDIA_OPENCL = -L/usr/local/cuda/lib64/
DIR_HEADER_NVIDIA_OPENCL = -I/usr/local/cuda/include/

clean:
	rm -f *.o
	rm -f $(TARGET)

intel:
	$(CC) -o $(TARGET) $(OBJECT) $(LIB_NAMES) $(DIR_LIB_INTEL_OPENCL) $(DIR_HEADER_INTEL_OPENCL)
	
nvidia: 
	$(CC) -o $(TARGET) $(OBJECT) $(LIB_NAMES) $(DIR_LIB_NVIDIA_OPENCL) $(DIR_HEADER_NVIDIA_OPENCL)
