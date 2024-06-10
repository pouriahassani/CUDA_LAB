#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <pthread.h>
#include <string.h>

#define BLOCK_SIZE 16

typedef struct
{
    float *A;
    float *B;
    float *C;
    int N;
    int startRow;
    int endRow;
} ThreadData;

void *MatrixMulCPUThread(void *arg)
{
    ThreadData *data = (ThreadData *)arg;
    float *A = data->A;
    float *B = data->B;
    float *C = data->C;
    int N = data->N;
    int startRow = data->startRow;
    int endRow = data->endRow;

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

    pthread_exit(NULL);
}

void MatrixMulCPU(float *A, float *B, float *C, int N, int numThreads)
{
    pthread_t *threads = (pthread_t *)malloc(numThreads * sizeof(pthread_t));
    ThreadData *threadData = (ThreadData *)malloc(numThreads * sizeof(ThreadData));

    int rowsPerThread = N / numThreads;
    int leftoverRows = N % numThreads;
    int startRow, endRow;

    for (int i = 0; i < numThreads; i++)
    {
        startRow = i * rowsPerThread;
        endRow = (i + 1) * rowsPerThread + (i == numThreads - 1 ? leftoverRows : 0);

        threadData[i].A = A;
        threadData[i].B = B;
        threadData[i].C = C;
        threadData[i].N = N;
        threadData[i].startRow = startRow;
        threadData[i].endRow = endRow;

        pthread_create(&threads[i], NULL, MatrixMulCPUThread, (void *)&threadData[i]);
    }

    // Wait for all threads to finish
    for (int i = 0; i < numThreads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    free(threadData);
}

// Figure source: https://users.wfu.edu/choss/CUDA/docs/Lecture 5.pdf

// Without shared memory, the performance is still promising because:
// 1. A large number of streaming multiprocessors can handle parallel threads simultaneously.
// 2. In matrix multiplication, each thread can independently compute one element of the result matrix C,
//    which improves computational efficiency.
// 3. Although global memory access is slower than shared memory, the memory bandwidth is still higher than that of CPUs.
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

// The block size is defined as 16, which means each thread block contains 16×16=256 threads.
__global__ void MatrixMulKernelTiled(float *A, float *B, float *C, int N)
{
    // Define shared memory tile_A and tile_B to store tiles of A and B.
    __shared__ float tile_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_B[BLOCK_SIZE][BLOCK_SIZE];

    // Calculate the global row and column indices in the result matrix C.
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Initialize sum variable to accumulate the result.
    float sum = 0;

    // Iterate over all tiles of A and B. Here, m represents the current tile index.
    for (int m = 0; m < N / BLOCK_SIZE; m++)
    {
        // Load the current tile of A and B from global memory to shared memory tile_A and tile_B.
        // Each thread loads one element of A and B.
        // For tile_A: each thread loads one element from the corresponding row of A, at position A[row * N + (m * BLOCK_SIZE + threadIdx.x)].
        // For tile_B: each thread loads one element from the corresponding column of B, at position B[(m * BLOCK_SIZE + threadIdx.y) * N + col].
        tile_A[threadIdx.y][threadIdx.x] = A[row * N + (m * BLOCK_SIZE + threadIdx.x)];
        tile_B[threadIdx.y][threadIdx.x] = B[(m * BLOCK_SIZE + threadIdx.y) * N + col];

        // Use __syncthreads() to ensure all threads have completed loading.
        __syncthreads();

        // Each thread performs the computation for the tile in shared memory.
        // By iterating over all elements within the tile, compute the partial results and accumulate them into sum.
        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        // Use __syncthreads() again to ensure all threads have completed the computation for the current tile.
        __syncthreads();
    }

    // Write the accumulated result to the corresponding position in the result matrix C.
    C[row * N + col] = sum;
}

double timespec_to_ms(struct timespec *ts)
{
    return ts->tv_sec * 1000.0 + ts->tv_nsec / 1000000.0;
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <NUM_THREADS> <SIZE>\n", argv[0]);
        return -1;
    }

    int numThreads = atoi(argv[1]);
    int N = atoi(argv[2]);

    FILE *f = fopen("res.csv", "a");

    fseek(f, 0, SEEK_END);
    if (ftell(f) == 0)
    {
        fprintf(f, "Threads,Size,CPU Time(ms),CUDA Time(ms),Tiled CUDA Time(ms)\n");
    }

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

    // clock_gettime() provides higher resolution time measurement and allows access to different clocks.
    // CLOCK_MONOTONIC is a monotonically increasing clock that is not affected by system time adjustments,
    // making it suitable for measuring time intervals.
    // The return value type is struct timespec, which includes seconds (tv_sec) and nanoseconds (tv_nsec),
    // offers more precise and high-resolution time measurement.

    // multi-threaded CPU
    struct timespec start_cpu, end_cpu;
    clock_gettime(CLOCK_MONOTONIC, &start_cpu);
    MatrixMulCPU(h_A, h_B, h_C_CPU, N, numThreads);
    clock_gettime(CLOCK_MONOTONIC, &end_cpu);
    double cpu_time_used = timespec_to_ms(&end_cpu) - timespec_to_ms(&start_cpu);

    cudaEvent_t start, stop, start_tiled, stop_tiled;
    float milliseconds = 0, milliseconds_tiled = 0;

    // Many CUDA functions are asynchronous; that is, they return control to the calling CPU thread
    // before the work is completed. All kernel launches are asynchronous, and memory copy
    // functions with the Async suffix are also asynchronous. Therefore, to accurately measure the
    // time consumed by a specific call or sequence of CUDA calls, the CPU thread must be
    // synchronized with the GPU by calling cudaDeviceSynchronize() immediately before
    // starting and stopping the CPU timer. cudaDeviceSynchronize() blocks the calling CPU
    // thread until all CUDA calls previously issued by that thread have completed.

    // Here, cudaEventRecord() is used to place the start and stop events into the default
    // stream. When it reaches the event in the stream, the device records the event's timestamp.
    // The cudaEventElapsedTime() function returns the time parameter, which is the GPU
    // computation time. This value is expressed in milliseconds, with a resolution of approximately
    // half a microsecond.

    // More in Nvidia’s documentation: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html

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
        if (fabs(h_C[i] - h_C_CPU[i]) > 1e-3)
        {
            error = true;
            break;
        }
    }

    if (!error)
    {
        fprintf(f, "%d,%d,%f,%f,%f\n", numThreads, N, cpu_time_used, milliseconds, milliseconds_tiled);
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

    fclose(f);

    return 0;
}
