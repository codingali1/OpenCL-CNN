__kernel void convolution_1(__global float *inputs,
                            __global float *conv_temp,
                            int D1, int N, int input_offset) {
    int g_i = get_global_id(0);
    int b_i = get_global_id(1);

    if(g_i >= (N * N * D1)) return;

    __global float *input = inputs + input_offset + (N * N * D1) * b_i;
    __global float *temp = conv_temp + (N * N * D1 * 3 * 3) * b_i;

    int i = g_i / N;
    int j = g_i % N;

    int channel = i / N;
    int channel_i = i - channel * N;

    for(int l = 0; l < 3; l++) {
        for(int k = 0; k < 3; k++) {
            int x = j + k - 1;
            int y = channel_i + l - 1;

            if (x >= 0 && x < N && y >= 0 && y < N)
                temp[(((channel * 3 * 3) + (3 * l + k)) * (N * N)) + (channel_i * N + j)] = input[((channel * N) + y) * N + x];
            else
                temp[(((channel * 3 * 3) + (3 * l + k)) * (N * N)) + (channel_i * N + j)] = 0.0f;
        }
    }
}

// _tiling_more_work_and_more_batch_per_thread
__kernel void convolution_2(
        __global float *conv_temp,
        __global float *networks,
        __global float *outputs,
        int D2, int D1, int N, int filter_offset
    )
{
    int b_i = get_global_id(2);

    const int ROW_A = D2;
    const int COL_A = D1 * 3 * 3;
    const int ROW_B = D1 * 3 * 3;
    const int COL_B = N * N;

    __global float *temp = conv_temp + (ROW_B * COL_B) * b_i * DEPTH;
    __global float *filter = networks + filter_offset;
    __global float *biases = networks + filter_offset + (ROW_A * COL_A);
    __global float *output = outputs + (ROW_A * COL_B) * b_i * DEPTH;

    __local float Asub[TS][TS];
    __local float Bsub[DEPTH][TS][TS];

    int li = get_local_id(1);
    int lj = get_local_id(0);
    int gi = get_group_id(1) * TS + li;
    int gj = get_group_id(0) * TS + lj;

    const int RTS = TS / WPT;

    float sum[DEPTH][WPT] = {{0.0f}};

    for (int t = 0; t < COL_A; t += TS) {
        for (int w = 0; w < WPT; w++) {
            const int tj = t + lj;
            const int ti = t + li;

            if (gi < ROW_A && (tj + (w * RTS)) < COL_A) {
                Asub[li][lj + (w * RTS)] = filter[gi * COL_A + tj + (w * RTS)];
            } else {
                Asub[li][lj + (w * RTS)] = 0.0f;
            }
        }

        for (int d = 0; d < DEPTH; d++) {
            for (int w = 0; w < WPT; w++) {
                const int tj = t + lj;
                const int ti = t + li;

                if (ti < ROW_B && (gj + (w * RTS)) < COL_B) {
                    Bsub[d][li][lj + (w * RTS)] = temp[(d * ROW_B * COL_B) + ti * COL_B + gj + (w * RTS)];
                } else {
                    Bsub[d][li][lj + (w * RTS)] = 0.0f;
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for(int d = 0; d < DEPTH; d++) {
            for (int k = 0; k < TS; k++) {
                for (int w = 0; w < WPT; w++) {
                    sum[d][w] += Asub[li][k] * Bsub[d][k][lj + (w * RTS)];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for(int d = 0; d < DEPTH; d++) {
        for (int w = 0; w < WPT; w++) {
            if (gi < ROW_A && (gj + (w * RTS)) < COL_B) {
                output[(d * ROW_A * COL_B) + gi * COL_B + gj + (w * RTS)] = ReLU(sum[d][w] + biases[gi]);
            }
        }
    }
}

// _tiling_more_batch_per_thread
__kernel void convolution_2_tiling_more_batch_per_thread(
        __global float *conv_temp,
        __global float *networks,
        __global float *outputs,
        int D2, int D1, int N, int filter_offset
    )
{
    int b_i = get_global_id(2);

    const int ROW_A = D2;
    const int COL_A = D1 * 3 * 3;
    const int ROW_B = D1 * 3 * 3;
    const int COL_B = N * N;

    __global float *temp = conv_temp + (ROW_B * COL_B) * b_i * DEPTH;
    __global float *filter = networks + filter_offset;
    __global float *biases = networks + filter_offset + (ROW_A * COL_A);
    __global float *output = outputs + (ROW_A * COL_B) * b_i * DEPTH;

    __local float Asub[TS][TS];
    __local float Bsub[DEPTH][TS][TS];

    int li = get_local_id(1);
    int lj = get_local_id(0);
    int gi = get_group_id(1) * TS + li;
    int gj = get_group_id(0) * TS + lj;

    float sum[DEPTH] = {{0.0f}};

    for (int t = 0; t < COL_A; t += TS) {
        const int tj = t + lj;
        const int ti = t + li;

        if (gi < ROW_A && tj < COL_A) {
            Asub[li][lj] = filter[gi * COL_A + tj];
        } else {
            Asub[li][lj] = 0.0f;
        }

        for (int d = 0; d < DEPTH; d++) {
            if (ti < ROW_B && gj < COL_B) {
                Bsub[d][li][lj] = temp[(d * ROW_B * COL_B) + ti * COL_B + gj];
            } else {
                Bsub[d][li][lj] = 0.0f;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for(int d = 0; d < DEPTH; d++) {
            for (int k = 0; k < TS; k++) {
                sum[d] += Asub[li][k] * Bsub[d][k][lj];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for(int d = 0; d < DEPTH; d++) {
        if (gi < ROW_A && gj < COL_B) {
            output[(d * ROW_A * COL_B) + gi * COL_B + gj] = ReLU(sum[d] + biases[gi]);
        }
    }
}

__kernel void pooling(__global float *inputs,
                      __global float *outputs,
                      int D, int N) {
    int g_i = get_global_id(0);
    int b_i = get_global_id(1);

    if(g_i >= (D * N * N)) return;

    __global float *input = inputs + (b_i * N * N * D * 4);
    __global float *output = outputs + (b_i * N * N * D);

    int i = g_i / (N * N);
    int m = (g_i / N) % N;
    int n = g_i % N;

    float max = 0.0f;

    for(int k = 0; k < 2; k++)
        for(int l = 0; l < 2; l++) {
            float pixel = input[(i * N * N * 4) + ((m * 2 + k) * 2 * N + n * 2 + l)];
            max = (max > pixel) ? max : pixel;
        }

    output[(i * N * N) + (m * N + n)] = max;
}

// _naive
__kernel void fc(__global float *input_neurons,
                 __global float *output_neurons,
                 __global float *networks,
                 int M, int N, int networks_offset) {
    int g_i = get_global_id(0);
    int b_i = get_global_id(1);

    if(g_i >= M) return;

    __global float *input_neuron = input_neurons + (b_i * N);
    __global float *output_neuron = output_neurons + (b_i * M);
    __global float *filter = networks + networks_offset;
    __global float * biases = networks + networks_offset + (M * N);

    float sum = 0.0f;

    for (int i = 0; i < N; i++)
        sum += input_neuron[i] * filter[g_i * N + i];

    sum += biases[g_i];
    output_neuron[g_i] = ReLU(sum);
}
