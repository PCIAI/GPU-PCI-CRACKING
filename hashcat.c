#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>

__global__ void add_matrices(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  int n = 1024;

  float *a, *b, *c;

  // Allocate memory on the host
  a = (float *)malloc(sizeof(float) * n);
  b = (float *)malloc(sizeof(float) * n);
  c = (float *)malloc(sizeof(float) * n);

  // Initialize the matrices
  for (int i = 0; i < n; i++) {
    a[i] = i;
    b[i] = i * 2;
  }

  // Allocate memory on the device
  cudaMalloc((void **)&a_device, sizeof(float) * n);
  cudaMalloc((void **)&b_device, sizeof(float) * n);
  cudaMalloc((void **)&c_device, sizeof(float) * n);

  // Copy the matrices to the device
  cudaMemcpy(a_device, a, sizeof(float) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(b_device, b, sizeof(float) * n, cudaMemcpyHostToDevice);

  // Launch the kernel
  add_matrices<<<1024, 1>>>(a_device, b_device, c_device, n);

  // Copy the result back to the host
  cudaMemcpy(c, c_device, sizeof(float) * n, cudaMemcpyDeviceToHost);

  // Print the result
  for (int i = 0; i < n; i++) {
    printf("%f\n", c[i]);
  }

  // Free the memory
  free(a);
  free(b);
  free(c);

  // Shutdown CUDA
  cudaFree(a_device);
  cudaFree(b_device);
  cudaFree(c_device);

  return 0;
}
