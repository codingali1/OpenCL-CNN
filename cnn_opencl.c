#include <CL/cl.h>
#include "cnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BATCH 	1000	// batch size
#define TS 		16		// tile size
#define WPT 	2		// work per thread
#define DEPTH 	4		// batch per thread

cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;
char* kernel_source;
size_t kernel_source_size;
cl_kernel kernel_convolution_1, kernel_convolution_2;
cl_kernel kernel_pooling, kernel_fc;
cl_int err;

cl_mem m_images;
cl_mem m_networks;
cl_mem m_conv_input, m_conv_temp, m_conv_output;
int input_offset, filter_offset;
cl_mem m_pooling_input, m_pooling_output;
cl_mem m_fc_input, m_fc_output;

size_t global_size[3] = { NULL, BATCH , 1};
size_t local_size[3] = { 256, 1, 1 };

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

char* get_source_code(const char* file_name, size_t* len) {
	char* source_code;
	char buf[2] = "\0";
	int cnt = 0;
	size_t length;
	FILE* file = fopen(file_name, "r");

	if (file == NULL) {
		printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}

	fseek(file, 0, SEEK_END);
	length = (size_t)ftell(file);
	rewind(file);
	
	source_code = (char*)malloc(length + 1);
	fread(source_code, length, 1, file);

	for (int i = 0; i < length; i++) {
		buf[0] = source_code[i];
		if (buf[0] == '\n') {
			cnt++;
		}
	}

	fclose(file);

	return source_code;
}

/*
 * D = channel size
 * N = width and height of an output image
 * Thus, input is (D, N * 2, N * 2) and output is (D, N, N).
 */
static void pooling_layer(cl_mem* inputs, cl_mem* outputs, int D, int N) {
	err = clSetKernelArg(kernel_pooling, 0, sizeof(cl_mem), inputs);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_pooling, 1, sizeof(cl_mem), outputs);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_pooling, 2, sizeof(int), &D);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_pooling, 3, sizeof(int), &N);
	CHECK_ERROR(err);

	global_size[0] = D * N * N;
	global_size[0] = (global_size[0] + local_size[0] - 1) / local_size[0] * local_size[0];

	err = clEnqueueNDRangeKernel(queue, kernel_pooling, 2, NULL, global_size, local_size, 0, NULL, NULL);
	CHECK_ERROR(err);
}

/*
 * D2 = output channel size
 * D1 = input channel size
 * N = width and height of an input image
 * input image is zero-padded by 1.
 * Thus, input is (D1, N, N) and output is (D2, N, N)
 */
static void convolution_layer(cl_mem* inputs, cl_mem* outputs, cl_mem* networks, int D2, int D1, int N) {
	// Convolution_1
	err = clSetKernelArg(kernel_convolution_1, 0, sizeof(cl_mem), inputs);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_convolution_1, 1, sizeof(cl_mem), &m_conv_temp);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_convolution_1, 2, sizeof(int), &D1);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_convolution_1, 3, sizeof(int), &N);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_convolution_1, 4, sizeof(int), &input_offset);
	CHECK_ERROR(err);

	global_size[0] = N * N * D1;
	global_size[0] = (global_size[0] + local_size[0] - 1) / local_size[0] * local_size[0];

	err = clEnqueueNDRangeKernel(queue, kernel_convolution_1, 2, NULL, global_size, local_size, 0, NULL, NULL);
	CHECK_ERROR(err);

	// Convolution_2
	int tile_size = 16;

	err = clSetKernelArg(kernel_convolution_2, 0, sizeof(cl_mem), &m_conv_temp);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_convolution_2, 1, sizeof(cl_mem), networks);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_convolution_2, 2, sizeof(cl_mem), outputs);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_convolution_2, 3, sizeof(int), &D2);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_convolution_2, 4, sizeof(int), &D1);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_convolution_2, 5, sizeof(int), &N);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_convolution_2, 6, sizeof(int), &filter_offset);
	CHECK_ERROR(err);

	// tiling: more work and more batch per thread
	global_size[0] = N * N / WPT;	global_size[1] = D2;	global_size[2] = BATCH / DEPTH;
	local_size[0] = TS / WPT;		local_size[1] = TS;		local_size[2] = 1;

	// tiling: more batch per thread
	// global_size[0] = N * N;	global_size[1] = D2;	global_size[2] = BATCH / DEPTH;
	// local_size[0] = TS;		local_size[1] = TS;		local_size[2] = 1;
	

	global_size[0] = (global_size[0] + local_size[0] - 1) / local_size[0] * local_size[0];
	global_size[1] = (global_size[1] + local_size[1] - 1) / local_size[1] * local_size[1];

	err = clEnqueueNDRangeKernel(queue, kernel_convolution_2, 3, NULL, global_size, local_size, 0, NULL, NULL);
	CHECK_ERROR(err);

	global_size[1] = BATCH; global_size[2] = 1;
	local_size[0] = 256;	local_size[1] = 1;
}

/*
 * M = output size
 * N = input size
 */
static void fc_layer(cl_mem* input_neuron, cl_mem* output_neuron, cl_mem* networks, int M, int N) {
	err = clSetKernelArg(kernel_fc, 0, sizeof(cl_mem), input_neuron);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_fc, 1, sizeof(cl_mem), output_neuron);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_fc, 2, sizeof(cl_mem), networks);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_fc, 3, sizeof(int), &M);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_fc, 4, sizeof(int), &N);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_fc, 5, sizeof(int), &filter_offset);
	CHECK_ERROR(err);

	global_size[0] = M;
	global_size[0] = (global_size[0] + local_size[0] - 1) / local_size[0] * local_size[0];

	err = clEnqueueNDRangeKernel(queue, kernel_fc, 2, NULL, global_size, local_size, 0, NULL, NULL);
	CHECK_ERROR(err);
}

static void softmax(float* output, int N) {
	int i;
	float max = output[0];
	for (i = 1; i < N; i++) {
		max = (output[i] > max) ? output[i] : max;
	}
	float sum = 0;
	for (i = 0; i < N; i++) {
		sum += exp(output[i] - max);
	}
	for (i = 0; i < N; i++) {
		output[i] = exp(output[i] - max) / sum;
	}
}

static int find_max(float* fc, int N) {
	int i;
	int maxid = 0;
	float maxval = 0;
	for (i = 0; i < N; i++) {
		if (maxval < fc[i]) {
			maxval = fc[i];
			maxid = i;
		}
	}
	return maxid;
}

float* alloc_layer(size_t n) {
	return (float*)malloc(n * sizeof(float));
}

void cnn_init() {

	 // Platform ID
	err = clGetPlatformIDs(1, &platform, NULL);
	CHECK_ERROR(err);

	// Device ID
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	CHECK_ERROR(err);

	// Create Context
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);

	// Create Command Queue
	queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
	CHECK_ERROR(err);

	// Create Program Object
	kernel_source = get_source_code("kernel.cl", &kernel_source_size);
	program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
	CHECK_ERROR(err);

	// Build Program
	char option[100];
	sprintf(option, "-cl-fast-relaxed-math -D ReLU(x)=(((x)>0)?(x):0) -D TS=%d -D WPT=%d -D DEPTH=%d", TS, WPT, DEPTH);
	err = clBuildProgram(program, 1, &device, option, NULL, NULL);
	if (err == CL_BUILD_PROGRAM_FAILURE) {
		size_t log_size;
		char* log;

		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		CHECK_ERROR(err);

		log = (char*)malloc(log_size + 1);
		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		CHECK_ERROR(err);

		log[log_size] = '\0';
		printf("Compiler error:\n%s\n", log);
		free(log);
		exit(0);
	};
	CHECK_ERROR(err);

	// Create Kernel
	kernel_convolution_1 = clCreateKernel(program, "convolution_1", &err);
	CHECK_ERROR(err);
	kernel_convolution_2 = clCreateKernel(program, "convolution_2", &err);
	CHECK_ERROR(err);
	kernel_pooling = clCreateKernel(program, "pooling", &err);
	CHECK_ERROR(err);
	kernel_fc = clCreateKernel(program, "fc", &err);
	CHECK_ERROR(err);
}

void cnn(float* images, float** network, int* labels, float* confidences, int num_images) {

	// Create Buffer
	m_images = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 32 * 32 * 3 * num_images, NULL, &err);
	CHECK_ERROR(err);
	m_networks = clCreateBuffer(context, CL_MEM_READ_ONLY, 60980520, NULL, &err);
	CHECK_ERROR(err);

	m_conv_input = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 32 * 32 * 64 * BATCH, NULL, &err);
	CHECK_ERROR(err);
	m_conv_temp = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 32 * 32 * 64 * 3 * 3 * BATCH, NULL, &err);
	CHECK_ERROR(err);
	m_conv_output = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 32 * 32 * 64 * BATCH, NULL, &err);
	CHECK_ERROR(err);

	m_pooling_input = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 32 * 32 * 64 * BATCH, NULL, &err);
	CHECK_ERROR(err);
	m_pooling_output = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 16 * 16 * 64 * BATCH, NULL, &err);
	CHECK_ERROR(err);

	m_fc_input = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 512 * BATCH, NULL, &err);
	CHECK_ERROR(err);
	m_fc_output = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 512 * BATCH, NULL, &err);
	CHECK_ERROR(err);

	// Write Buffer
	err = clEnqueueWriteBuffer(queue, m_images, CL_FALSE, 0, sizeof(float) * 32 * 32 * 3 * num_images, images, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, m_networks, CL_FALSE, 0, 60980520, *network, 0, NULL, NULL);
	CHECK_ERROR(err);

	float* fc3 = alloc_layer(10 * BATCH);

	// run network
	for (int i = 0; i < num_images / BATCH; ++i) {

		// Convolution Layer & Pooling Layer
		input_offset = i * 3 * 32 * 32 * BATCH; filter_offset = 0;
		convolution_layer(&m_images, &m_conv_output, &m_networks, 64, 3, 32);
		input_offset = 0;   filter_offset += (64 * 3 * 3 * 3) + 64;
		convolution_layer(&m_conv_output, &m_conv_input, &m_networks, 64, 64, 32);
		pooling_layer(&m_conv_input, &m_pooling_output, 64, 16);

		filter_offset += (64 * 64 * 3 * 3) + 64;
		convolution_layer(&m_pooling_output, &m_conv_output, &m_networks, 128, 64, 16);
		filter_offset += (128 * 64 * 3 * 3) + 128;
		convolution_layer(&m_conv_output, &m_conv_input, &m_networks, 128, 128, 16);
		pooling_layer(&m_conv_input, &m_pooling_output, 128, 8);

		filter_offset += (128 * 128 * 3 * 3) + 128;
		convolution_layer(&m_pooling_output, &m_conv_output, &m_networks, 256, 128, 8);
		filter_offset += (256 * 128 * 3 * 3) + 256;
		convolution_layer(&m_conv_output, &m_conv_input, &m_networks, 256, 256, 8);
		filter_offset += (256 * 256 * 3 * 3) + 256;
		convolution_layer(&m_conv_input, &m_conv_output, &m_networks, 256, 256, 8);
		pooling_layer(&m_conv_output, &m_pooling_output, 256, 4);

		filter_offset += (256 * 256 * 3 * 3) + 256;
		convolution_layer(&m_pooling_output, &m_conv_output, &m_networks, 512, 256, 4);
		filter_offset += (512 * 256 * 3 * 3) + 512;
		convolution_layer(&m_conv_output, &m_conv_input, &m_networks, 512, 512, 4);
		filter_offset += (512 * 512 * 3 * 3) + 512;
		convolution_layer(&m_conv_input, &m_conv_output, &m_networks, 512, 512, 4);
		pooling_layer(&m_conv_output, &m_pooling_output, 512, 2);

		filter_offset += (512 * 512 * 3 * 3) + 512;
		convolution_layer(&m_pooling_output, &m_conv_output, &m_networks, 512, 512, 2);
		filter_offset += (512 * 512 * 3 * 3) + 512;
		convolution_layer(&m_conv_output, &m_conv_input, &m_networks, 512, 512, 2);
		filter_offset += (512 * 512 * 3 * 3) + 512;
		convolution_layer(&m_conv_input, &m_conv_output, &m_networks, 512, 512, 2);
		pooling_layer(&m_conv_output, &m_pooling_output, 512, 1);

		// FC layer
		filter_offset += (512 * 512 * 3 * 3) + 512;
		fc_layer(&m_pooling_output, &m_fc_output, &m_networks, 512, 512);
		filter_offset += (512 * 512) + 512;
		fc_layer(&m_fc_output, &m_fc_input, &m_networks, 512, 512);
		filter_offset += (512 * 512) + 512;
		fc_layer(&m_fc_input, &m_fc_output, &m_networks, 10, 512);

		err = clEnqueueReadBuffer(queue, m_fc_output, CL_TRUE, 0, sizeof(float) * 10 * BATCH, fc3, 0, NULL, NULL);
		CHECK_ERROR(err);

		float* fc_3 = fc3;
		for (int j = 0; j < BATCH; j++) {
			// Softmax
			softmax(fc3, 10);

			// Find max
			labels[i * BATCH + j] = find_max(fc3, 10);
			confidences[i * BATCH + j] = fc3[labels[i * BATCH + j]];
			fc3 += 10;
		}
		fc3 = fc_3;
	}

	clReleaseMemObject(m_images); clReleaseMemObject(m_networks);
	clReleaseMemObject(m_conv_input); clReleaseMemObject(m_conv_temp); clReleaseMemObject(m_conv_output);
	clReleaseMemObject(m_pooling_input); clReleaseMemObject(m_pooling_output);
	clReleaseMemObject(m_fc_input); clReleaseMemObject(m_fc_output);
	clReleaseKernel(kernel_convolution_1); clReleaseKernel(kernel_convolution_2);
	clReleaseKernel(kernel_pooling); clReleaseKernel(kernel_fc);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	free(fc3);
}