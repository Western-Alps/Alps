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
__global__ void
transpose_weights( double *Weights, double *Weights_T, 
		   const int L_k, const int L_k_minus_1, 
		   const int Offset )
{
  //  __shared__ float tile[TILE_DIM * TILE_DIM];
  //  

  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  //printf( " (%d, %d) ",x , y );
  int w_position   = Offset + x*L_k_minus_1 + y;
  int w_T_position = Offset + x + y*L_k;
  //
  if( x < L_k && y <  L_k_minus_1 )
    {
      Weights_T[w_T_position] = Weights[w_position];
      printf(" -- Weights_T[%d] = %f", w_T_position, Weights_T[w_T_position]);
    }

//  int width = gridDim.x * TILE_DIM;
//  
//  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
//     tile[(threadIdx.y+j)*TILE_DIM + threadIdx.x] = idata[(y+j)*width + x];
//  
//  __syncthreads();
//  
//  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
//    odata[(y+j)*width + x] = tile[(threadIdx.y+j)*TILE_DIM + threadIdx.x];          
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
MAC::FullyConnected_layer_CUDA::init()
{
  if ( !CUDA_init_ )
    {
      //
      // Arguments
      std::cout << "CUDA FullyConnected_layer_CUDA init." << std::endl;
      const int Num_fc_layers = 4;
      int FC_layers[Num_fc_layers] = { 199, 3, 3, 2 };
      //
      int Numweights = 620;
      int Numneurons = 211;
      //
      double *Weights   = new double[Numweights];
      int count = 0;
      int weights_offset = 0;
      for ( int layer = 1 ; layer <  Num_fc_layers; layer++ )
	{
	  std::cout << FC_layers[layer] << "*" << FC_layers[layer-1]+1 << std::endl;
	  for ( int a = 0 ; a < FC_layers[layer] ; a++ )
	    for ( int n = 0 ; n < FC_layers[layer-1]+1 ; n++ )
	      {
		// W_a,n
		int w_position = weights_offset + a*(FC_layers[layer-1]+1) + n;
		Weights[w_position] = layer*10000 + a*1000 + n;
		std::cout << " w_pos: " << w_position 
			  << " W:" << Weights[w_position];
		count++;
	      }
	  std::cout << std::endl;
	  //
	  weights_offset += (FC_layers[layer-1]+1)*FC_layers[layer];
	}
      //
      std::cout << "count: " << count << std::endl;

      
      //
      // CUDA processing
      //
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess)
	{
	  fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
	  exit(EXIT_FAILURE);
	}
      
      //
      // 1. Allocate memory on the device
      err = cudaMalloc((void **)&d_weights_, Numweights* sizeof(double) );
      err = cudaMalloc((void **)&d_weights_T_, Numweights* sizeof(double) );
 

      //
      // 2. Copy on the device
      err = cudaMemcpy( d_weights_, Weights, Numweights * sizeof(double), cudaMemcpyHostToDevice );

      //
      // 3. Launch the kernel
      int threadsPerBlock = 32;
      weights_offset = 0;
      for ( int layer = 1 ; layer <  Num_fc_layers; layer++ )
	{
	  int 
	    L1 = ( FC_layers[layer]     + threadsPerBlock - 1) / threadsPerBlock,
	    L2 = ( FC_layers[layer-1]+1 + threadsPerBlock - 1) / threadsPerBlock;
	  dim3 dim_Block(threadsPerBlock, threadsPerBlock);
	  dim3 dim_Grid(L1, L2);


	  std::cout << FC_layers[layer-1]+1 << "*" << FC_layers[layer] << std::endl;
	  std::cout << dim_Grid.x << "," << dim_Grid.y << "," << dim_Grid.z << std::endl;
	  std::cout << dim_Block.x << "," << dim_Block.y << "," << dim_Block.z << std::endl;
	  transpose_weights<<< dim_Grid, dim_Block >>>( d_weights_,
							d_weights_T_,
							FC_layers[layer],
							FC_layers[layer-1]+1,
							weights_offset );
	  //
	  weights_offset += (FC_layers[layer-1]+1)*FC_layers[layer];
	}
      //
      if (err != cudaSuccess)
	{
	  fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
	  exit(EXIT_FAILURE);
	}
      
      //
      // Copy back the data for test
      double *weights_T = new double[Numweights];
      err = cudaMemcpy( weights_T, d_weights_T_, Numweights * sizeof(double), 
			cudaMemcpyDeviceToHost );
      if (err != cudaSuccess)
	{
	  fprintf(stderr, "Failed to copy from device to host (error code %s)!\n", 
		  cudaGetErrorString(err));
	  exit(EXIT_FAILURE);
	}
      
      for ( int i = 0 ; i < Numweights ; i++ )
	std::cout << weights_T[i] << " ";
      std::cout << std::endl;
      
      
      //
      // initialization done
      CUDA_init_ = true;
    }
}
//
//
//
__host__ void
MAC::FullyConnected_layer_CUDA::backward()
{
//  std::cout << "In FullyConnected_layer_CUDA" << std::endl;
//  init_();
//  const int num_fc_layers = 4;
//  int fc_layers[num_fc_layers] = { 2+1, 3+1, 3+1, 2 };
//  //
//  int numweights = 29;
//  int numneurons = 13;
//  //
//  double *weights   = new double[numweights];
//  double *weights_T = new double[numweights];
//  double *delta     = new double[numneurons];
//  double *z_l       = new double[numneurons];
//  double *z_ll      = new double[numneurons];
//  //
//  double *d_weights;
//  double *d_delta;
//  double *d_z_l;
//  double *d_z_ll;
//  //
//  for ( int i = 0 ; i < numweights ; i++ )
//    {
//      weights[i] = i;
//      std::cout << " i = " << i
//		<< " w = " << weights[i];
//      if ( i < numneurons )
//	{
//	  z_l[i] = z_ll[i] = i;
//	  delta[i] = 0.;
//	  std::cout << " delta = " << delta[i];
//	}
//    }
//  std::cout << std::endl;
//  delta[2+1 + 3+1 + 3+1     ] = 100.;
//  delta[2+1 + 3+1 + 3+1 + 1 ] = 200.;
//  
//  //
//  // CUDA processing
//  //
//  cudaError_t err = cudaGetLastError();
//  if (err != cudaSuccess)
//    {
//      fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
//      exit(EXIT_FAILURE);
//    }
//
//
//  //
//  // 1. Allocate memory on the device
//  err = cudaMalloc((void **)&d_weights, numweights* sizeof(double) );
//  err = cudaMalloc((void **)&d_delta,   numneurons* sizeof(double) );
//  err = cudaMalloc((void **)&d_z_l,     numneurons* sizeof(double) );
//  err = cudaMalloc((void **)&d_z_ll,    numneurons* sizeof(double) );
//
//  //
//  // 2. Copy on the device
//  err = cudaMemcpy( d_weights, weights, numweights * sizeof(double), cudaMemcpyHostToDevice );
//  err = cudaMemcpy( d_delta, delta,     numneurons * sizeof(double), cudaMemcpyHostToDevice );
//  err = cudaMemcpy( d_z_l, z_l,         numneurons * sizeof(double), cudaMemcpyHostToDevice );
//  err = cudaMemcpy( d_z_ll, z_ll,       numneurons * sizeof(double), cudaMemcpyHostToDevice );
//
//  //
//  // Check everythong went well
//  if (err != cudaSuccess)
//    {
//      fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
//      exit(EXIT_FAILURE);
//    }
// 
//  //
//  // 3. Launch the kernel
//  int threadsPerBlock = 256;
//  int blocksPerGrid =( numweights + threadsPerBlock - 1) / threadsPerBlock;
//  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
//  //
//
//
//  for ( int i = num_fc_layers - 1 ; i > 0 ; i-- )
//    {
//      int
//	offset_1 = 0,
//	offset_2 = 0;
//      for ( int j = 0 ; j < i ; j++ )
//	{
//	  offset_2 += fc_layers[j];
//	  if ( j < i-1 )
//	    offset_1 += fc_layers[j];
//	}
//
//      delta_cuda<<< blocksPerGrid, threadsPerBlock >>>( d_weights, d_delta,
//							numneurons - offset_2);
//      //sqrt_cuda<<< blocksPerGrid, threadsPerBlock >>>(d_weights, numweights );
//      std::cout << std::endl;
//      std::cout
//	<< " offset_1 :" << offset_1
//	<< " offset_2 :" << offset_2
//	<<  std::endl;
//    }
//  
//  
//  
//  if (err != cudaSuccess)
//    {
//      fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
//      exit(EXIT_FAILURE);
//    }
//
//  err = cudaMemcpy(weights, d_weights, numweights * sizeof(double), cudaMemcpyDeviceToHost);
//    if (err != cudaSuccess)
//    {
//        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
//        exit(EXIT_FAILURE);
//    }
//
//  for ( int i = 0 ; i < numweights ; i++ )
//    std::cout << weights[i] << " ";
//  std::cout << std::endl;
};
//
//
//
__host__ 
MAC::FullyConnected_layer_CUDA::~FullyConnected_layer_CUDA()
{
  cudaError_t err = cudaGetLastError();
  err = cudaFree( d_weights_ );
  err = cudaFree( d_weights_T_ );
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