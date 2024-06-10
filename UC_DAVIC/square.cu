///////////////////////////////////////////////////////////////////////////
// Example code from the Udacity tutorial on CUDA                        //
// Link to the video here: https://www.youtube.com/watch?v=GiGE3QjwknQ   //
// TO COMPILE: $ nvcc -o square square.cu                                //
///////////////////////////////////////////////////////////////////////////

//
// NOTES:
//  * Device: is the term of the GPU
//  * Host: is the term of the CPU
//  * kernels are the only things run in parallel on the GPU
//  * everything in the main is run on CPU
//  * memory transfers between Host (CPU) and Device (GPU) should be minimal
//  * kernels all run at the same time
//  * threads can know their Id's with threadIdx.x, blocks are similar
//

#include <stdio.h>

// kernel to be run on the TX2
__global__ void square(float *d_out, float *d_in){
  int idx = threadIdx.x; // this is how you get the thread index
  float f = d_in[idx];
  d_out[idx] = f*f;
}

// main is here. this is the CPU code. 
int main(){
  // the size of the array, which is really the thread count per block
  const int ARRAY_SIZE = 64; // max thread count per block on the TX2
  // total bytes in the array
  const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

  // Generate the input array on the HOST
  float h_in[ARRAY_SIZE];
  for (int i = 0; i < ARRAY_SIZE; i++){
    h_in[i] = float(i);
  }
  float h_out[ARRAY_SIZE]; // Define the output array on the HOST
  
  // Declare pointers to input and output arrays for the DEVICE
  float *d_in;
  float *d_out;

  //Allocate the memory on the DEVICE and assign the location of the allocated memory to the device pointers
  cudaMalloc((void **) &d_in, ARRAY_BYTES);
  cudaMalloc((void **) &d_out, ARRAY_BYTES);

  // Transfer the input array from HOST to the Device. "Copy's contents of h_in to d_in"
  cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

  // launch the kernel
  square<<<1,ARRAY_SIZE>>>(d_out,d_in);

  // copy the result back to the HOST from the DEVICE
  cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

  // print the result
  for (int i = 0; i < ARRAY_SIZE; i++){
    printf("%f", h_out[i]);
    printf(((i % 4) != 3) ? "\t" : "\n");
  }

  // free the allocated memory on the device 
  cudaFree(d_in);
  cudaFree(d_out);

  return 0;
}

