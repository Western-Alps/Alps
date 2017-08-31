//
//
//
#include "MACException.h"
#include "NN_test.cuh"

/**
 * CUDA Kernel Device code
 *
 * Computes ...
 */
__global__ void
sqrt_cuda( double *A, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
      
        A[i] = A[i] * A[i];
	/*printf("lalla %f", A[i]);*/
    }
}


//
//
//
MAC::NN_test::NN_test( double* Weights, double* D_Weights ):
  MAC::NeuralNetwork::NeuralNetwork(),
  weights_{Weights}, d_weights_{D_Weights}
{

};
//
//
//
__host__ void
MAC::NN_test::forward()
{
  std::cout << "In NN_test" << std::endl;
  for ( int i = 0 ; i < 2000 ; i++ )
    std::cout << weights_[i] << " ";
  std::cout << std::endl;

  std::cout << "CUDA processing" << std::endl;

  int threadsPerBlock = 256;
  int numElements = 2000;
  int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
  sqrt_cuda<<< blocksPerGrid, threadsPerBlock >>>(d_weights_, numElements);

  cudaError_t err = cudaGetLastError();
  
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
