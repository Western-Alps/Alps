//
//
//
#include "MACException.h"
#include "FullyConnected_layer_CUDA.cuh"
//
// layer orgnization of the densly connected network
// the data will be saved in the constant memory of the device
//
__constant__ int d_fc_layers_[20];
/**
 * CUDA Kernel Device code
 *
 * Computes the delta element of the backward processing
 */
__global__ void
delta_cuda( const double *Weights_T, double *Delta, const double *A,
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
	  //printf("Weights_T(%d) = %f \n", w_position+i, Weights_T[w_position+i]);
	}
      // 
      Delta[l2] *= delta_omega;
    }
}
/**
 * CUDA Kernel Device code
 *
 * Computes the gradient of the costfunction
 */
__global__ void
grad_E_cuda( double *grad_E , const double *Delta, const double *Z, 
	     const int Weights_offset,
	     const int OffSet2, const int Layer2,
	     const int OffSet3, const int Layer3  )
{
  //
  //
  int l2 = blockDim.x * blockIdx.x + threadIdx.x;
  int l3 = blockDim.y * blockIdx.y + threadIdx.y;
  //
  if ( l2 < d_fc_layers_[Layer2]+1 && l3 < d_fc_layers_[Layer3] )
    {
      int w_position      = Weights_offset + l3*(d_fc_layers_[Layer2]+1) + l2;
      grad_E[w_position] += Delta[OffSet3 + l3] * Z[OffSet2 + l2];
    }
}
/**
 * CUDA Kernel Device code
 *
 * Computes the uptade of the weights in the gradient descent
 */
__global__ void
gradient_descent_cuda( double *Weights, double *grad_E,
		       const double Eta, const int Size )
{
  //
  //
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  //
  if ( position < Size )
    {
      Weights[position] -= Eta * grad_E[position];
      grad_E[position]   = 0.0;
    }
}
/**
 * CUDA Kernel Device code
 *
 * Initialize an array on the device
 */
__global__ void
array_init_cuda( double *Array, const int Size )
{
  //
  //
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  //
  if ( position < Size )
    Array[position] = 0.0;
}
/**
 * CUDA Kernel Device code
 *
 * Computes the transposed weights matrices for backward processing
 */
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
{};
//
//
//
__host__ void
MAC::FullyConnected_layer_CUDA::init( const int  Number_fc_layers,
				      const int* Fc_layers,
				      const int Number_of_weights, 
				      const double* Weights )
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
      err = cudaMalloc((void **)&d_E_,         number_of_weights_* sizeof(double) );
      err = cudaMalloc((void **)&d_weights_,   number_of_weights_* sizeof(double) );
      err = cudaMalloc((void **)&d_weights_T_, number_of_weights_* sizeof(double) );
      // device
      int threadsPerBlock = 1024;
      int numBlocks = (( number_of_weights_ ) + threadsPerBlock - 1) / threadsPerBlock;
      // 3.1. Compute delta
      array_init_cuda<<< numBlocks, threadsPerBlock >>>( d_E_, number_of_weights_ );


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

  ////
  //// test
  //double *E_test = new double[number_of_weights_];
  //err = cudaMemcpy(E_test, d_E_, number_of_weights_ * sizeof(double), cudaMemcpyDeviceToHost);
  //if (err != cudaSuccess)
  //  {
  //    fprintf(stderr, "Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(err));
  //    exit(EXIT_FAILURE);
  //  }
  ////
  //std::cout << "Energy -- start" << std::endl;
  //for ( int i = 0 ; i < number_of_weights_ ; i++ )
  //  std::cout << E_test[i] << " ";
  //std::cout << std::endl;
  //std::cout << "Energy -- end" << std::endl;


  //
  // Loop over the subjects
  for ( std::map< std::string, Neurons_type >::iterator image = Neurons.begin() ;
	image != Neurons.end() ; image++ )
    {
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
	  fprintf(stderr, "Failed to allocate mem on device (error code %s)!\n",
		  cudaGetErrorString(err));
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
			    (std::get< 0/*activations*/>( (*image).second )[layer].get()),
			    size * sizeof(double), cudaMemcpyHostToDevice );
	  err = cudaMemcpy( (d_z_l + neuron_position),
			    (std::get< 1/*neurons*/>( (*image).second )[layer].get()), 
			    size * sizeof(double), cudaMemcpyHostToDevice );
	  err = cudaMemcpy( (d_delta + neuron_position),
			    (std::get< 2/*deltas*/>( (*image).second )[layer].get()),
			    size * sizeof(double), cudaMemcpyHostToDevice );
	  //
	  // Check everythong went well
	  if (err != cudaSuccess)
	    {
	      fprintf(stderr, "Failed to copy host to device (error code %s)!\n",
		      cudaGetErrorString(err));
	      exit(EXIT_FAILURE);
	    }

	  //
	  //
	  neuron_position += size;
	}

      //
      // 3. Launch the kernels to compute delta and the energy gradient
      int threadsPerBlock = 32;
      for ( int i = number_fc_layers_ - 1 ; i > 0 ; i-- )
	{
	  //
	  // neurons and weights offset
	  int
	    offset_1 = 0,
	    offset_2 = 0,
	    offset_3 = 0,
	    weights_offset_1 = 0,
	    weights_offset_2 = 0;
	  // neurons
	  for ( int j = 0 ; j < i ; j++ )
	    {
	      offset_3 += fc_layers_[j] + 1;
	      if ( j < i-1 )
		offset_2 += fc_layers_[j] + 1;
	      if ( j < i-2 )
		offset_1 += fc_layers_[j] + 1;
	    }
	  // weights
	  for ( int j = 1 ; j < i ; j++ )
	    {
	      weights_offset_2 += (fc_layers_[j-1]+1)*fc_layers_[j];
	      if ( j < i-1 )
		weights_offset_1 += (fc_layers_[j-1]+1)*fc_layers_[j];
	    }
	  //
	  std::cout
	    << " offset_1 :" << offset_1
	    << " weights_offset_1: " << weights_offset_1
	    << " weights_offset_2: " << weights_offset_2
	    << " offset_2 :" << offset_2
	    << " offset_3 :" << offset_3
	    <<  std::endl;
	      
	  //
	  //
	  int 
	    L2 = ((fc_layers_[i-1] + 1) + threadsPerBlock - 1) / threadsPerBlock,
	    L3 = ((fc_layers_[i] ) + threadsPerBlock - 1) / threadsPerBlock;
	  //
	  // 3.1. Compute delta
	  delta_cuda<<< L2, threadsPerBlock >>>( d_weights_T_, d_delta, d_a_l,
						 weights_offset_2,
						 offset_2, i-1,
						 offset_3, i );
	  //
	  if (err != cudaSuccess)
	    {
	      fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
	      exit(EXIT_FAILURE);
	    }
	  //
	  // 3.2. Compute gradient E_
	  dim3 dim_Block(threadsPerBlock, threadsPerBlock);
	  dim3 dim_Grid(L2, L3);
	  //
	  grad_E_cuda<<< dim_Grid, dim_Block >>>( d_E_, d_delta, d_z_l, 
						  weights_offset_2,
						  offset_2, i-1,
						  offset_3, i );
	  //
	  if (err != cudaSuccess)
	    {
	      fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
	      exit(EXIT_FAILURE);
	    }

	  ////
	  //// test
	  //double *delta = new double[number_of_neurons_];
	  //err = cudaMemcpy(delta, d_delta, number_of_neurons_ * sizeof(double),
	  //		   cudaMemcpyDeviceToHost);
	  //if (err != cudaSuccess)
	  //	{
	  //	  fprintf(stderr, "Failed to copy from device to host (error code %s)!\n",
	  //		  cudaGetErrorString(err));
	  //	  exit(EXIT_FAILURE);
	  //	}
	  ////
	  //std::cout << "Backward step -- start" << std::endl;
	  //for ( int ii = 0 ; ii < number_of_neurons_ ; ii++ )
	  //	std::cout << delta[ii] << " ";
	  //std::cout << std::endl;
	  //std::cout << "Backward step -- end" << std::endl;
	}
      
      //
      // 5. Download the error \delta_{k}
      neuron_position = 0;
      for ( int layer = 0 ; layer < number_fc_layers_ ; layer++ )
	{
	  //
	  //
	  int size = fc_layers_[layer] + ( layer == number_fc_layers_ - 1 ? 0 : 1 );
	  //
	  err = cudaMemcpy( (std::get< 0/*activations*/>( (*image).second )[layer].get()),
			    (d_a_l + neuron_position),
			    size * sizeof(double), cudaMemcpyDeviceToHost );
	  err = cudaMemcpy( (std::get< 1/*neurons*/>( (*image).second )[layer].get()), 
			    (d_z_l + neuron_position),
			    size * sizeof(double), cudaMemcpyDeviceToHost );
	  err = cudaMemcpy( (std::get< 2/*deltas*/>( (*image).second )[layer].get()),
			    (d_delta + neuron_position),
			    size * sizeof(double), cudaMemcpyDeviceToHost );
	  //
	  // Check everythong went well
	  if (err != cudaSuccess)
	    {
	      fprintf(stderr, "Failed to copy host to device (error code %s)!\n",
		      cudaGetErrorString(err));
	      exit(EXIT_FAILURE);
	    }
	  
	  //
	  //
	  neuron_position += size;
	}

      //
      // 6. free the space
      err = cudaFree( d_a_l );
      err = cudaFree( d_z_l );
      err = cudaFree( d_delta );
      if (err != cudaSuccess)
	{
	  fprintf( stderr, "Failed to free device vector C (error code %s)!\n",
		  cudaGetErrorString(err) );
	  exit(EXIT_FAILURE);
	}
    }
  

  //
  // 4. Update the weights with the gradient descent
  std::cout << "Compute the new weights." << std::endl;
  // and re-initialise the energy to 0
  int threadsPerBlock = 1024;
  int numBlocks = (( number_of_weights_ ) + threadsPerBlock - 1) / threadsPerBlock;
  // 3.1. Compute delta
  gradient_descent_cuda<<< numBlocks, threadsPerBlock >>>( d_weights_, d_E_,
							   learning_rate_,
							   number_of_weights_ );
  //  //
  //  // test
  //  double *E_test = new double[number_of_weights_];
  //  err = cudaMemcpy(E_test, d_weights_, number_of_weights_ * sizeof(double),
  //  		   cudaMemcpyDeviceToHost);
  //  if (err != cudaSuccess)
  //    {
  //      fprintf(stderr, "Failed to copy from device to host (error code %s)!\n",
  //  	      cudaGetErrorString(err));
  //      exit(EXIT_FAILURE);
  //    }
  //  //
  //  std::cout << "Energy -- start" << std::endl;
  //  for ( int ii = 0 ; ii < number_of_weights_ ; ii++ )
  //    std::cout << std::fixed << E_test[ii] << " ";
  //  std::cout << std::endl;
  //  std::cout << "Energy -- end" << std::endl;
};
//
//
//
__host__ 
MAC::FullyConnected_layer_CUDA::~FullyConnected_layer_CUDA()
{
  cudaError_t err = cudaGetLastError();
  err = cudaFree( d_E_ );
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