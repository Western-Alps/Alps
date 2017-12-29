//
//
//
#include "MACException.h"
#include "FullyConnected_layer_CUDA.cuh"

__constant__ int d_fc_layers_[20];
/**
 * CUDA Kernel Device code
 *
 * Computes ...
 */
__global__ void
delta_cuda( const double *Weights_T, double *Delta,
	    const double *Z, const double *A,
	    const int Weights_offset,
	    const int OffSet2, const int Layer2,
	    const int OffSet3, const int Layer3,
	    const bool last_layer )
{
  //
  //
  int postion = blockDim.x * blockIdx.x + threadIdx.x;
  int l2 = OffSet2 + postion;
  //
  if ( l2 < OffSet3 )
    {
      //Delta[l2] = 1 - tanh( A[l2] ) * tanh( A[l2] );
      Delta[l2] = A[l2] ;
      // layer l_{u+1}
      double delta_omega = 0.;
      int w_position = Weights_offset + postion * d_fc_layers_[Layer3];
      for ( int i = 0 ; i < d_fc_layers_[Layer3] ; i++ )
	{
	  delta_omega += Delta[OffSet3 + i] * Weights_T[w_position + i];
	  printf("Delta(%d) = %f \n", OffSet3 + i, Delta[OffSet3 + i]);
	  printf("Weights_T(%d) = %f \n", w_position+i, Weights_T[w_position+i]);
	}
    }
}
__global__ void
grad_E_cuda( const double *Weights_T, double *Delta,
	     const double *Z, const double *A,
	     const int OffSet1, const int Layer1,
	     const int OffSet2, const int Layer2,
	     const int OffSet3, const int Layer3,
	     const int Number_of_neurons)
{
  //
  //
  int l1 = OffSet1 + blockDim.x * blockIdx.x + threadIdx.x;
  int l2 = OffSet2 + blockDim.y * blockIdx.y + threadIdx.y;
  //
  if ( l1 < OffSet2 &&  l2 < OffSet3 )
    {
      printf("(%d,%d,%d) ", d_fc_layers_[Layer1],d_fc_layers_[Layer2],d_fc_layers_[Layer3]);
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
  //
  //
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  //
  // 
  int w_position   = Offset + x*L_k_minus_1 + y;
  int w_T_position = Offset + x + y*L_k;
  //
  if( x < L_k && y <  L_k_minus_1 )
    Weights_T[ w_T_position ] = Weights[ w_position ];
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
      // !!!!Arguments
      std::cout << "CUDA FullyConnected_layer_CUDA init." << std::endl;
      const int Num_fc_layers = 4;
      int FC_layers[Num_fc_layers] = { 9, 3, 3, 2 };
      //
      int Numweights = 50;
      int Numneurons = 20;
      //
      double *Weights   = new double[Numweights];
      int count = 0;
      int weights_offset = 0;
      for ( int layer = 1 ; layer <  Num_fc_layers; layer++ )
	{
	  for ( int a = 0 ; a < FC_layers[layer] ; a++ )
	    for ( int n = 0 ; n < FC_layers[layer-1]+1 ; n++ )
	      {
		// W_a,n
		int w_position = weights_offset + a*(FC_layers[layer-1]+1) + n;
		Weights[w_position] = layer*10000 + a*1000 + n;
		count++;
	      }
	  //
	  weights_offset += (FC_layers[layer-1]+1)*FC_layers[layer];
	}
      //
      // !!!!Arguments
      
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
      err = cudaMemcpyToSymbol( d_fc_layers_, FC_layers, Num_fc_layers * sizeof(int) );

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

      
      ////
      //// Copy back the data for test
      //double *weights_T = new double[Numweights];
      //err = cudaMemcpy( weights_T, d_weights_T_, Numweights * sizeof(double), 
      //			cudaMemcpyDeviceToHost );
      //if (err != cudaSuccess)
      //	{
      //	  fprintf(stderr, "Failed to copy from device to host (error code %s)!\n", 
      //		  cudaGetErrorString(err));
      //	  exit(EXIT_FAILURE);
      //	}
      //
      //for ( int i = 0 ; i < Numweights ; i++ )
      //	std::cout << weights_T[i] << " ";
      //std::cout << std::endl;
      
      
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
  //
  //
  std::cout << "In FullyConnected_layer_CUDA" << std::endl;
  init();

  //
  // !!! Arguments
  const int Num_fc_layers = 4;
  int FC_layers[Num_fc_layers] = { 9, 3, 3, 2 };
  //
  int Numweights = 50;
  int Numneurons = 20;
  //
  double *delta     = new double[Numneurons];
  double *z_l       = new double[Numneurons];
  double *a_l       = new double[Numneurons];
  //
  double *d_delta;
  double *d_z_l;
  double *d_a_l;
  //
  for ( int i = 0 ; i < Numweights ; i++ )
    {
      if ( i < Numneurons )
	{
	  z_l[i] = i;
	  a_l[i] = i;
	  delta[i] = 0.;
	}
    }
  delta[Numneurons - 2] = 100.;
  delta[Numneurons - 1] = 200.;
  //
  // !!! Arguments
  
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
  err = cudaMalloc((void **)&d_delta,   Numneurons* sizeof(double) );
  err = cudaMalloc((void **)&d_z_l,     Numneurons* sizeof(double) );
  err = cudaMalloc((void **)&d_a_l,     Numneurons* sizeof(double) );

  //
  // 2. Copy on the device
  err = cudaMemcpy( d_delta, delta,     Numneurons * sizeof(double), cudaMemcpyHostToDevice );
  err = cudaMemcpy( d_z_l, z_l,         Numneurons * sizeof(double), cudaMemcpyHostToDevice );
  err = cudaMemcpy( d_a_l, a_l,         Numneurons * sizeof(double), cudaMemcpyHostToDevice );

  //
  // Check everythong went well
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to copy host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
 
  //
  // 3. Launch the kernel
  int threadsPerBlock = 32;
  for ( int i = Num_fc_layers - 1 ; i > 0 ; i-- )
    {
      //
      // neurons and weights offset
      int
	offset_1 = 0,
	offset_2 = 0,
	offset_3 = 0,
	weights_offset = 0;
      // neurons
      for ( int j = 0 ; j < i ; j++ )
	{
	  offset_3 += FC_layers[j] + 1;
	  if ( j < i-1 )
	    offset_2 += FC_layers[j] + 1;
	  //if ( j < i-2 )
	  //  offset_1 += FC_layers[j] + 1;
	}
      // weights
      for ( int j = 1 ; j < i ; j++ )
	weights_offset += (FC_layers[j-1]+1)*FC_layers[j];
      
      //
      //
      int 
	L1 = ((offset_2 - offset_1) + threadsPerBlock - 1) / threadsPerBlock,
	L2 = ((offset_3 - offset_2) + threadsPerBlock - 1) / threadsPerBlock;
      //
      std::cout << "weights_offset: " << weights_offset << std::endl;
      delta_cuda<<< L2, threadsPerBlock >>>( d_weights_T_, d_delta, 
					     d_z_l, d_a_l,
					     weights_offset,
					     offset_2, i-1,
					     offset_3, i,
					     i == Num_fc_layers - 1 );
      std::cout << std::endl;
      std::cout
	<< " offset_1 :" << offset_1
	<< " offset_2 :" << offset_2
	<< " offset_3 :" << offset_3
	<<  std::endl;
    }
  
  //
  //
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to launch  kernel (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

  err = cudaMemcpy(delta, d_delta, Numneurons * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

  for ( int i = 0 ; i < Numneurons ; i++ )
    std::cout << delta[i] << " ";
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