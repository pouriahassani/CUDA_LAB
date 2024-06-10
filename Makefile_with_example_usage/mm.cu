#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <thread>
#include <vector>

#define BLOCK_SIZE 16

void MatrixMulCPUThread(float *A, float *B, float *C, int N, int startRow, int endRow)
{
    for (int row = startRow; row < endRow; row++)
    {
        for (int col = 0; col < N; col++)
        {
            float sum = 0.0f;
            for (int n = 0; n < N; n++)
            {
                sum += A[row * N + n] * B[n * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

void MatrixMulCPU(float *A, float *B, float *C, int N, int numThreads)
{
    std::vector<std::thread> threads;

    int rowsPerThread = N / numThreads;
    int leftoverRows = N % numThreads;
    int startRow, endRow;

    for (int i = 0; i < numThreads; i++)
    {
        startRow = i * rowsPerThread;
        endRow = (i + 1) * rowsPerThread + (i == numThreads - 1 ? leftoverRows : 0);

        threads.emplace_back(MatrixMulCPUThread, A, B, C, N, startRow, endRow);
    }

    // wait all threads finish
    for (auto &thr : threads)
    {
        thr.join();
    }
}

__global__ void MatrixMulKernel(float *A, float *B, float *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;

    if (row < N && col < N)
    {
        for (int i = 0; i < N; i++)
        {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void MatrixMulKernelTiled(float *A, float *B, float *C, int N)
{
    __shared__ float tile_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_B[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0;

    // iterate tile
    for (int m = 0; m < N / BLOCK_SIZE; m++)
    {
        // load tile to shared memory
        tile_A[threadIdx.y][threadIdx.x] = A[row * N + (m * BLOCK_SIZE + threadIdx.x)];
        tile_B[threadIdx.y][threadIdx.x] = B[(m * BLOCK_SIZE + threadIdx.y) * N + col];
        __syncthreads();

        // compute
        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * N + col] = sum;
}

int main()
{
    FILE *f = fopen("mm.csv", "w");

    fprintf(f, "Size,CPU Time(ms),CUDA Time(ms),Tiled CUDA Time(ms)\n");

    int numThreads = std::thread::hardware_concurrency();

    for (int N = 64; N <= 2048; N *= 2)
    {
        size_t size = N * N * sizeof(float);

        float *h_A, *h_B, *h_C, *h_C_CPU;
        float *d_A, *d_B, *d_C;

        h_A = (float *)malloc(size);
        h_B = (float *)malloc(size);
        h_C = (float *)malloc(size);
        h_C_CPU = (float *)malloc(size);

        // initialize
        for (int i = 0; i < N * N; ++i)
        {
            h_A[i] = rand() / (float)RAND_MAX;
            h_B[i] = rand() / (float)RAND_MAX;
            h_C_CPU[i] = 0.0f;
        }

        cudaMalloc(&d_A, size);
        cudaMalloc(&d_B, size);
        cudaMalloc(&d_C, size);

        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);

        // multi-threaded CPU
        clock_t start_cpu = clock();
        MatrixMulCPU(h_A, h_B, h_C_CPU, N, numThreads);
        clock_t end_cpu = clock();
        double cpu_time_used = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000;

        cudaEvent_t start, stop, start_tiled, stop_tiled;
        float milliseconds = 0, milliseconds_tiled = 0;

        // CUDA
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        MatrixMulKernel<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Tiled CUDA
        cudaEventCreate(&start_tiled);
        cudaEventCreate(&stop_tiled);
        cudaEventRecord(start_tiled);
        MatrixMulKernelTiled<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop_tiled);
        cudaEventSynchronize(stop_tiled);
        cudaEventElapsedTime(&milliseconds_tiled, start_tiled, stop_tiled);

        // copy result from GPU to host
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

        bool error = false;
        for (int i = 0; i < N * N; ++i)
        {
            if (fabs(h_C[i] - h_C_CPU[i]) > 1e-3) // error if set as 1e-5
            {
                error = true;
                break;
            }
        }

        if (!error)
        {
            fprintf(f, "%d,%f,%f,%f\n", N, cpu_time_used, milliseconds, milliseconds_tiled);
        }
        else
        {
            printf("Error in multiplication at size %d\n", N);
        }

        free(h_A);
        free(h_B);
        free(h_C);
        free(h_C_CPU);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaEventDestroy(start_tiled);
        cudaEventDestroy(stop_tiled);
    }

    fclose(f);

    return 0;
}