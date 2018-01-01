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
	    const int OffSet3, const int Layer3 )
{
  //
  //
  int postion = blockDim.x * blockIdx.x + threadIdx.x;
  int l2 = OffSet2 + postion;
  //
  if ( l2 < OffSet3 )
    {
      Delta[l2] = 1 - tanh( A[l2] ) * tanh( A[l2] );
      //Delta[l2] = A[l2] ;
      // layer l_{u+1}
      double delta_omega = 0.;
      int w_position = Weights_offset + postion * d_fc_layers_[Layer3];
      for ( int i = 0 ; i < d_fc_layers_[Layer3] ; i++ )
	{
	  delta_omega += Delta[OffSet3 + i] * Weights_T[w_position + i];
	  //printf("Delta(%d) = %f \n", OffSet3 + i, Delta[OffSet3 + i]);
	  //printf("Weights_T(%d) = %f \n", w_position+i, Weights_T[w_position+i]);
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
  //printf ("Je passe");
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
MAC::FullyConnected_layer_CUDA::init( const int  Number_fc_layers,
				      const int* Fc_layers,
				      const int Number_of_weights, 
				      const double * Weights )
{
  if ( !CUDA_init_ )
    {
      //
      // Initialization
      std::cout << "CUDA FullyConnected_layer_CUDA init." << std::endl;
      number_fc_layers_ = Number_fc_layers;
      //
      fc_layers_ = new int[Number_fc_layers];
      memcpy( fc_layers_, Fc_layers, Number_fc_layers*sizeof(int) );
      //
      number_of_weights_ = Number_of_weights;
      // Compute number of neurons
      for ( int layer = 0 ; layer < number_fc_layers_ ; layer++ )
	number_of_neurons_ += fc_layers_[layer] + ( layer == number_fc_layers_ - 1 ? 0 : 1 );

      //
      // Allocate memory on host and device
      //

      //
      // Init the device
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess)
	{
	  fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
	  exit(EXIT_FAILURE);
	}

      //
      // 1. Allocate memory on the device
      err = cudaMalloc((void **)&d_weights_,   number_of_weights_* sizeof(double) );
      err = cudaMalloc((void **)&d_weights_T_, number_of_weights_* sizeof(double) );
      // host
      weights_ = new double[Number_of_weights];
      

      //
      // 2. Copy on the device
      err = cudaMemcpy( d_weights_, Weights, number_of_weights_ * sizeof(double), 
			cudaMemcpyHostToDevice );
      err = cudaMemcpyToSymbol( d_fc_layers_, fc_layers_, number_fc_layers_ * sizeof(int) );

      //
      // initialization done
      CUDA_init_ = true;
    }
}
//
//
//
__host__ void
MAC::FullyConnected_layer_CUDA::transpose_weight_matrices()
{
  //
  //
  std::cout << "CUDA FullyConnected_layer_CUDA transpose weights matrices." << std::endl;
  // Init the device
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
      
  //
  // 1. Launch the kernel
  int threadsPerBlock = 32;
  int weights_offset = 0;
  for ( int layer = 1 ; layer <  number_fc_layers_; layer++ )
    {
      int 
	L1 = ( fc_layers_[layer]     + threadsPerBlock - 1) / threadsPerBlock,
	L2 = ( fc_layers_[layer-1]+1 + threadsPerBlock - 1) / threadsPerBlock;
      dim3 dim_Block(threadsPerBlock, threadsPerBlock);
      dim3 dim_Grid(L1, L2);
      //
      transpose_weights<<< dim_Grid, dim_Block >>>( d_weights_,
						    d_weights_T_,
						    fc_layers_[layer],
						    fc_layers_[layer-1]+1,
						    weights_offset );
      //
      weights_offset += (fc_layers_[layer-1]+1)*fc_layers_[layer];
    }
  //
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

      
  ////
  //// Copy back the data for test
  //double *weights_T = new double[number_of_weights_];
  //err = cudaMemcpy( weights_T, d_weights_T_, number_of_weights_ * sizeof(double), 
  //		    cudaMemcpyDeviceToHost );
  //if (err != cudaSuccess)
  //	{
  //	  fprintf(stderr, "Failed to copy from device to host (error code %s)!\n", 
  //		  cudaGetErrorString(err));
  //	  exit(EXIT_FAILURE);
  //	}
  ////
  //std::cout 
  //  << "number_of_weights_ " <<number_of_weights_
  //  << std::endl;
  //for ( int i = 0 ; i < number_of_weights_ ; i++ )
  //	std::cout << "weights_T[" << i << "] = " << weights_T[i] << " ";
  //std::cout << std::endl;
}
//
//
//
__host__ void
MAC::FullyConnected_layer_CUDA::backward( std::map< std::string, Neurons_type >& Neurons )
{
  //
  //
  std::cout << "In FullyConnected_layer_CUDA Backpropagation." << std::endl;
  // CUDA processing
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }


  //
  // Loop over the subjects
  for ( std::map< std::string, Neurons_type >::iterator image = Neurons.begin() ;
  image != Neurons.end() ; image++ )
    {
      std::cout << (*image).first << std::endl;
      //
      //
      double 
	*d_delta, 
	*d_z_l, 
	*d_a_l;

      //
      // 1. Allocate memory on the device
      err = cudaMalloc((void **)&d_delta, number_of_neurons_* sizeof(double) );
      err = cudaMalloc((void **)&d_z_l,   number_of_neurons_* sizeof(double) );
      err = cudaMalloc((void **)&d_a_l,   number_of_neurons_* sizeof(double) );
      //
      // Check everythong went well
      if (err != cudaSuccess)
	{
	  fprintf(stderr, "Failed to copy host to device (error code %s)!\n", cudaGetErrorString(err));
	  exit(EXIT_FAILURE);
	}

      //
      // 2. copy the data on the device
      int neuron_position = 0;
      for ( int layer = 0 ; layer < number_fc_layers_ ; layer++ )
	{
	  //
	  //
	  int size = fc_layers_[layer] + ( layer == number_fc_layers_ - 1 ? 0 : 1 );
	  //
	  err = cudaMemcpy( (d_a_l + neuron_position),
			    (std::get< 0/*activations*/>( (*image).second )[layer].get() + neuron_position),
			    size * sizeof(double), cudaMemcpyHostToDevice );
	  err = cudaMemcpy( (d_z_l + neuron_position),
			    (std::get< 1/*neurons*/>( (*image).second )[layer].get() + neuron_position), 
			    size * sizeof(double), cudaMemcpyHostToDevice );
	  err = cudaMemcpy( (d_delta + neuron_position),
			    (std::get< 2/*deltas*/>( (*image).second )[layer].get() + neuron_position),
			    size * sizeof(double), cudaMemcpyHostToDevice );
	  //
	  // Check everythong went well
	  if (err != cudaSuccess)
	    {
	      fprintf(stderr, "Failed to copy host to device (error code %s)!\n", cudaGetErrorString(err));
	      exit(EXIT_FAILURE);
	    }
	  //
	  neuron_position += size;
	}

      //
      // 3. Launch the kernel
      int threadsPerBlock = 32;
      for ( int i = number_fc_layers_ - 1 ; i > 0 ; i-- )
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
	      offset_3 += fc_layers_[j] + 1;
	      if ( j < i-1 )
		offset_2 += fc_layers_[j] + 1;
	      //if ( j < i-2 )
	      //  offset_1 += fc_layers_[j] + 1;
	    }
	  // weights
	  for ( int j = 1 ; j < i ; j++ )
	    weights_offset += (fc_layers_[j-1]+1)*fc_layers_[j];
	  
	  //
	  //
	  int 
	    //L1 = ((offset_2 - offset_1) + threadsPerBlock - 1) / threadsPerBlock,
	    L2 = ((offset_3 - offset_2) + threadsPerBlock - 1) / threadsPerBlock;
	  //
	  std::cout << "weights_offset: " << weights_offset << std::endl;
	  delta_cuda<<< L2, threadsPerBlock >>>( d_weights_T_, d_delta, 
						 d_z_l, d_a_l,
						 weights_offset,
						 offset_2, i-1,
						 offset_3, i );
	  std::cout << std::endl;
	  std::cout
	    << " offset_1 :" << offset_1
	    << " offset_2 :" << offset_2
	    << " offset_3 :" << offset_3
	    <<  std::endl;
	}
      //
      if (err != cudaSuccess)
	{
	  fprintf(stderr, "Failed to launch  kernel (error code %s)!\n", cudaGetErrorString(err));
	  exit(EXIT_FAILURE);
	}

      //
      // test
      double *delta = new double[number_of_neurons_];
      err = cudaMemcpy(delta, d_delta, number_of_neurons_ * sizeof(double), cudaMemcpyDeviceToHost);
      if (err != cudaSuccess)
	{
	  fprintf(stderr, "Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(err));
	  exit(EXIT_FAILURE);
	}
      
      for ( int i = 0 ; i < number_of_neurons_ ; i++ )
	std::cout << delta[i] << " ";
      std::cout << std::endl;
    }
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