//
//
//
#include "MACException.h"
#include "FullyConnected_layer_CUDA.cuh"

/**
 * CUDA Kernel Device code
 *
 * Computes ...
 */
__global__ void
sqrt_cuda( double *A, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if ( i < numElements )
    {
        A[i] = A[i] * A[i];
	printf("A[%d] = %f ", i, A[i]);
    }
}


//
//
//
MAC::FullyConnected_layer_CUDA::FullyConnected_layer_CUDA()
{
  weights_ = new double[2000];
};
//
//
//
__host__ void
MAC::FullyConnected_layer_CUDA::forward()
{
  std::cout << "In FullyConnected_layer_CUDA" << std::endl;
  int numElements = 2000;
  for ( int i = 0 ; i < numElements ; i++ )
    {
      weights_[i] = i*i;
      std::cout << weights_[i] << " ";
    }
  std::cout << std::endl;

  cudaError_t err = cudaGetLastError();

  err = cudaMalloc((void **)&d_weights_, numElements * sizeof(double) );
  
  err = cudaMemcpy( d_weights_, weights_, numElements * sizeof(double), cudaMemcpyHostToDevice );
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
 

  std::cout << "CUDA processing" << std::endl;

  int threadsPerBlock = 256;
  int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
  sqrt_cuda<<< blocksPerGrid, threadsPerBlock >>>(d_weights_, numElements);

  
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(weights_, d_weights_, numElements * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

  for ( int i = 0 ; i < 2000 ; i++ )
    std::cout << weights_[i] << " ";
  std::cout << std::endl;
};
//
//
//
__host__ 
MAC::FullyConnected_layer_CUDA::~FullyConnected_layer_CUDA()
{
  cudaError_t err = cudaGetLastError();
  err = cudaFree( d_weights_ );
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}