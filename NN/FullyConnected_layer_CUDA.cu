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
delta_cuda( double *Weights,
	    double *Delta,
	    int Last_layer_size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if ( i < Last_layer_size + 3 )
    {
        printf(" delta[%d] = %f ", i, Delta[i]);
    }
}
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
};
//
//
//
__host__ void
MAC::FullyConnected_layer_CUDA::forward()
{
  std::cout << "In FullyConnected_layer_CUDA" << std::endl;
  const int num_fc_layers = 4;
  int fc_layers[num_fc_layers] = { 2+1, 3+1, 3+1, 2 };
  //
  int numweights = 27;
  int numneurons = 13;
  //
  double *weights = new double[numweights];
  double *delta  = new double[numneurons];
  double *z_l    = new double[numneurons];
  double *z_ll   = new double[numneurons];
  //
  double *d_weights;
  double *d_delta;
  double *d_z_l;
  double *d_z_ll;
  //
  for ( int i = 0 ; i < numweights ; i++ )
    {
      weights[i] = i;
      std::cout << " i = " << i
		<< " w = " << weights[i];
      if ( i < numneurons )
	{
	  z_l[i] = z_ll[i] = i;
	  delta[i] = 0.;
	  std::cout << " delta = " << delta[i];
	}
    }
  std::cout << std::endl;
  delta[2+1 + 3+1 + 3+1     ] = 100.;
  delta[2+1 + 3+1 + 3+1 + 1 ] = 200.;
  
  //
  // CUDA processing
  //
  cudaError_t err = cudaGetLastError();

  //
  // 1. Allocate memory on the device
  err = cudaMalloc((void **)&d_weights, numweights* sizeof(double) );
  err = cudaMalloc((void **)&d_delta,   numneurons* sizeof(double) );
  err = cudaMalloc((void **)&d_z_l,     numneurons* sizeof(double) );
  err = cudaMalloc((void **)&d_z_ll,    numneurons* sizeof(double) );

  //
  // 2. Copy on the device
  err = cudaMemcpy( d_weights, weights, numweights * sizeof(double), cudaMemcpyHostToDevice );
  err = cudaMemcpy( d_delta, delta,     numneurons * sizeof(double), cudaMemcpyHostToDevice );
  err = cudaMemcpy( d_z_l, z_l,         numneurons * sizeof(double), cudaMemcpyHostToDevice );
  err = cudaMemcpy( d_z_ll, z_ll,       numneurons * sizeof(double), cudaMemcpyHostToDevice );

  //
  // Check everythong went well
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
 
  //
  // 3. Launch the kernel
  int threadsPerBlock = 256;
  int blocksPerGrid =( numweights + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
  //


  for ( int i = num_fc_layers - 1 ; i > 0 ; i-- )
    {
      int
	offset_1 = 0,
	offset_2 = 0;
      for ( int j = 0 ; j < i ; j++ )
	{
	  offset_2 += fc_layers[j];
	  if ( j < i-1 )
	    offset_1 += fc_layers[j];
	}

      delta_cuda<<< blocksPerGrid, threadsPerBlock >>>( d_weights, d_delta,
							numneurons - offset_2);
      //sqrt_cuda<<< blocksPerGrid, threadsPerBlock >>>(d_weights, numweights );
      std::cout << std::endl;
      std::cout
	<< " offset_1 :" << offset_1
	<< " offset_2 :" << offset_2
	<<  std::endl;
    }
  
  
  
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

  err = cudaMemcpy(weights, d_weights, numweights * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

  for ( int i = 0 ; i < numweights ; i++ )
    std::cout << weights[i] << " ";
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