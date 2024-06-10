// #include <stdio.h>
// #include <cuda_runtime.h>

// __global__ void no_divergence_kernel(int *data, int num_elements)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (idx < num_elements)
//     {
//         data[idx] = data[idx] * 2;
//     }
// }

// __global__ void divergence_kernel(int *data, int num_elements)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (idx < num_elements)
//     {
//         if (idx % 2 == 0)
//         {
//             data[idx] = data[idx] * 2;
//         }
//         else
//         {
//             // change operation to create divergence
//             data[idx] = data[idx] / 2;
//         }
//     }
// }

// int main()
// {
//     const int num_elements = 1024 * 1024;
//     size_t size = num_elements * sizeof(int);
//     int *h_data = (int *)malloc(size);
//     int *d_data;
//     cudaMalloc(&d_data, size);

//     // Fill data by host
//     for (int i = 0; i < num_elements; ++i)
//     {
//         h_data[i] = i;
//     }

//     // Copy data to GPU
//     cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

//     int threads_per_block = 256;
//     int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

//     // No divergence
//     cudaEvent_t start_no_div, stop_no_div;
//     float time_no_div = 0;
//     cudaEventCreate(&start_no_div);
//     cudaEventCreate(&stop_no_div);
//     cudaEventRecord(start_no_div);
//     no_divergence_kernel<<<blocks_per_grid, threads_per_block>>>(d_data, num_elements);
//     cudaEventRecord(stop_no_div);
//     cudaEventSynchronize(stop_no_div);
//     cudaEventElapsedTime(&time_no_div, start_no_div, stop_no_div);

//     cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

//     // With divergence
//     cudaEvent_t start_div, stop_div;
//     float time_div = 0;
//     cudaEventCreate(&start_div);
//     cudaEventCreate(&stop_div);
//     cudaEventRecord(start_div);
//     divergence_kernel<<<blocks_per_grid, threads_per_block>>>(d_data, num_elements);
//     cudaEventRecord(stop_div);
//     cudaEventSynchronize(stop_div);
//     cudaEventElapsedTime(&time_div, start_div, stop_div);

//     printf("No divergence time: %f ms\n", time_no_div);
//     printf("With divergence time: %f ms\n", time_div);

//     cudaEventDestroy(start_no_div);
//     cudaEventDestroy(stop_no_div);
//     cudaEventDestroy(start_div);
//     cudaEventDestroy(stop_div);
//     cudaFree(d_data);
//     free(h_data);
//     cudaDeviceReset();

//     return 0;
// }

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void nondivergent_kernel(int *a, int *b, int *c, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void divergent_kernel(int *a, int *b, int *c, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        if (idx % 2 == 0)
        {
            c[idx] = a[idx] + b[idx];
        }
        else
        {
            c[idx] = a[idx] * b[idx];
        }
    }
}

int main()
{
    const int N = 1024 * 1024; // array size
    int *a, *b, *c;

    // allocate device memory
    cudaMallocManaged(&a, N * sizeof(int));
    cudaMallocManaged(&b, N * sizeof(int));
    cudaMallocManaged(&c, N * sizeof(int));

    // initialize arrays
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i;
    }

    const int threads_per_block = 256;
    const int block_count = (N + threads_per_block - 1) / threads_per_block;

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    // non-divergent kernel
    cudaEventRecord(start_event);
    nondivergent_kernel<<<block_count, threads_per_block>>>(a, b, c, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    float milliseconds_nondivergent = 0;
    cudaEventElapsedTime(&milliseconds_nondivergent, start_event, stop_event);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    // divergent kernel
    cudaEventRecord(start_event);
    divergent_kernel<<<block_count, threads_per_block>>>(a, b, c, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    float milliseconds_divergent = 0;
    cudaEventElapsedTime(&milliseconds_divergent, start_event, stop_event);

    printf("Time for non-divergent kernel: %f ms\n", milliseconds_nondivergent);
    printf("Time for divergent kernel: %f ms\n", milliseconds_divergent);

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}