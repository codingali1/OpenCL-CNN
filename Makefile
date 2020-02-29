CC 		= gcc
CFLAGS	= -O2
OBJECT	= OpenCL_CNN.o cnn_opencl.o Compare_result.o
TARGET	= cnn_opencl

LIB_NAMES = -lOpenCL -lm
DIR_LIB_INTEL_OPENCL = -L/opt/intel/system_studio_2020/opencl/SDK/lib64/
DIR_HEADER_INTEL_OPENCL = -I/opt/intel/system_studio_2020/opencl/SDK/include/

all: $(TARGET)

clean:
	rm -f *.o
	rm -f $(TARGET)

$(TARGET): $(OBJECT)
	$(CC) -o $(TARGET) $(OBJECT) $(LIB_NAMES) $(DIR_LIB_INTEL_OPENCL) $(DIR_HEADER_INTEL_OPENCL)