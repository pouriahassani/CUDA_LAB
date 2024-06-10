#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define BLOCK_SIZE 256

typedef struct
{
    int *row_indices;
    int *col_indices;
    float *values;
    int num_nonzeros;
    int num_rows;
    int num_cols;
} COOMatrix;

__global__ void SpMVKernelCOO(int *d_row_indices, int *d_col_indices, float *d_values, float *d_x, float *d_y, int num_nonzeros)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_nonzeros)
    {
        atomicAdd(&d_y[d_row_indices[i]], d_values[i] * d_x[d_col_indices[i]]);
    }
}

__global__ void SpMVKernelNaive(float *d_A, float *d_x, float *d_y, int num_rows, int num_cols)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_rows)
    {
        float dot_product = 0;
        for (int col = 0; col < num_cols; col++)
        {
            dot_product += d_A[row * num_cols + col] * d_x[col];
        }
        d_y[row] = dot_product;
    }
}

void generateRandomCOOMatrix(COOMatrix *cooMatrix, int num_rows, int num_cols, int num_nonzeros)
{
    cooMatrix->num_rows = num_rows;
    cooMatrix->num_cols = num_cols;
    cooMatrix->num_nonzeros = num_nonzeros;
    cooMatrix->row_indices = (int *)malloc(num_nonzeros * sizeof(int));
    cooMatrix->col_indices = (int *)malloc(num_nonzeros * sizeof(int));
    cooMatrix->values = (float *)malloc(num_nonzeros * sizeof(float));

    for (int i = 0; i < num_nonzeros; i++)
    {
        cooMatrix->row_indices[i] = rand() % num_rows;
        cooMatrix->col_indices[i] = rand() % num_cols;
        cooMatrix->values[i] = (float)(rand() % 100) / 10.0;
    }
}

void generateRandomDenseMatrix(float *A, int num_rows, int num_cols)
{
    for (int i = 0; i < num_rows * num_cols; i++)
    {
        A[i] = (float)(rand() % 100) / 10.0;
    }
}

void freeCOOMatrix(COOMatrix *cooMatrix)
{
    free(cooMatrix->row_indices);
    free(cooMatrix->col_indices);
    free(cooMatrix->values);
}

void SpMVCOO(int *d_row_indices, int *d_col_indices, float *d_values, float *d_x, float *d_y, int num_nonzeros, cudaStream_t stream, float *kernel_time)
{
    int numBlocks = (num_nonzeros + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);

    SpMVKernelCOO<<<numBlocks, BLOCK_SIZE, 0, stream>>>(d_row_indices, d_col_indices, d_values, d_x, d_y, num_nonzeros);

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(kernel_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void SpMVNaive(float *d_A, float *d_x, float *d_y, int num_rows, int num_cols, cudaStream_t stream, float *kernel_time)
{
    int numBlocks = (num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);

    SpMVKernelNaive<<<numBlocks, BLOCK_SIZE, 0, stream>>>(d_A, d_x, d_y, num_rows, num_cols);

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(kernel_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

double get_time_in_ms()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s <num_rows> <num_cols> <num_nonzeros>\n", argv[0]);
        return -1;
    }

    int num_rows = atoi(argv[1]);
    int num_cols = atoi(argv[2]);
    int num_nonzeros = atoi(argv[3]);

    srand(time(NULL));

    // Generate random matrices
    COOMatrix cooMatrix;
    generateRandomCOOMatrix(&cooMatrix, num_rows, num_cols, num_nonzeros);

    float *A = (float *)malloc(num_rows * num_cols * sizeof(float));
    generateRandomDenseMatrix(A, num_rows, num_cols);

    float *x = (float *)malloc(num_cols * sizeof(float));
    float *y = (float *)malloc(num_rows * sizeof(float));
    float *y_naive = (float *)malloc(num_rows * sizeof(float));
    for (int i = 0; i < num_cols; i++)
    {
        x[i] = (float)(rand() % 100) / 10.0;
    }

    int *d_row_indices, *d_col_indices;
    float *d_values, *d_A, *d_x, *d_y;

    cudaMalloc((void **)&d_row_indices, cooMatrix.num_nonzeros * sizeof(int));
    cudaMalloc((void **)&d_col_indices, cooMatrix.num_nonzeros * sizeof(int));
    cudaMalloc((void **)&d_values, cooMatrix.num_nonzeros * sizeof(float));
    cudaMalloc((void **)&d_A, num_rows * num_cols * sizeof(float));
    cudaMalloc((void **)&d_x, num_cols * sizeof(float));
    cudaMalloc((void **)&d_y, num_rows * sizeof(float));

    cudaMemcpy(d_row_indices, cooMatrix.row_indices, cooMatrix.num_nonzeros * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, cooMatrix.col_indices, cooMatrix.num_nonzeros * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, cooMatrix.values, cooMatrix.num_nonzeros * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A, num_rows * num_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, num_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, num_rows * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float kernel_time_coo = 0;
    float kernel_time_naive = 0;
    double start_time, end_time, execution_time_coo, execution_time_naive;

    // COO version
    start_time = get_time_in_ms();
    SpMVCOO(d_row_indices, d_col_indices, d_values, d_x, d_y, cooMatrix.num_nonzeros, stream, &kernel_time_coo);
    cudaStreamSynchronize(stream);
    end_time = get_time_in_ms();
    execution_time_coo = end_time - start_time;

    cudaMemcpyAsync(y, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Naive version
    cudaMemset(d_y, 0, num_rows * sizeof(float)); // Reset the output vector for the naive version

    start_time = get_time_in_ms();
    SpMVNaive(d_A, d_x, d_y, num_rows, num_cols, stream, &kernel_time_naive);
    cudaStreamSynchronize(stream);
    end_time = get_time_in_ms();
    execution_time_naive = end_time - start_time;

    cudaMemcpyAsync(y_naive, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Validate results
    for (int i = 0; i < 10; i++)
    {
        printf("y_coo[%d] = %f, y_naive[%d] = %f\n", i, y[i], i, y_naive[i]);
    }

    FILE *f = fopen("res.csv", "a");
    if (f == NULL)
    {
        fprintf(stderr, "Error opening file for writing\n");
        return -1;
    }
    fprintf(f, "%d,%d,%d,%f,%f,%f,%f\n", num_rows, num_cols, num_nonzeros, execution_time_coo, kernel_time_coo, execution_time_naive, kernel_time_naive);
    fclose(f);

    free(x);
    free(y);
    free(y_naive);
    free(A);
    freeCOOMatrix(&cooMatrix);

    cudaFree(d_row_indices);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

    cudaStreamDestroy(stream);

    return 0;
}
