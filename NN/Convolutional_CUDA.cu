//
//
//
#include "MACException.h"
#include "Convolutional_CUDA.cuh"
#include "Activations.h"
//
#define THREADSPERBLOCK 1024
////
//// Note that any atomic operation can be implemented based on atomicCAS() (Compare And Swap). For example, atomicAdd() for double-precision floating-point numbers is not available on devices with compute capability lower than 6.0 but it can be implemented as follows: 
//#if __CUDA_ARCH__ < 600
//__device__ double atomicAdd_patch(double* address, double val)
//{
//    unsigned long long int* address_as_ull =
//                              (unsigned long long int*)address;
//    unsigned long long int old = *address_as_ull, assumed;
//
//    do {
//        assumed = old;
//        old = atomicCAS(address_as_ull, assumed,
//                        __double_as_longlong(val +
//                               __longlong_as_double(assumed)));
//
//    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
//    } while (assumed != old);
//
//    return __longlong_as_double(old);
//}
//#endif
////
////
//__global__ void
//test_cuda( double **tempo, MAC::Mapping* Map, int mod )
//{
//  printf("image(%d) = %f \n", mod, tempo[mod][421578]);
//  printf("Map(%d) = [%d,%d,%d] \n", 421578, Map[421578].x_, Map[421578].y_, Map[421578].z_);
//}
//
//
//
__global__ void
convolution_cuda( int  Image_size_in,
		  int  Image_size_out,
		  int          Number_of_weights,
		  double*      To_conv,
		  double*      Conv,
		  double**     Shared_weights,
		  double*      Shared_biases,
		  int* Weights_pos_oi )
{
  //
  //
  int odx = blockDim.x * blockIdx.x + threadIdx.x;
  double conv = 0.;
  if ( odx <  Image_size_out )
    {
      if (odx == 0)
	for ( int k = 0 ; k < Number_of_weights ; k++ )
	  {
	    //conv += Shared_weights[0][k] * To_conv[ Weights_pos_oi[odx][k] ];
	    printf(" Odx: %d ~~ %d ", odx,  Weights_pos_oi[k] );
	  }
    }
  //
  Conv[odx] = conv;
}
//
//
//
__host__ 
MAC::Convolutional_CUDA::Convolutional_CUDA()
{
}
//
//
//
__host__ void
MAC::Convolutional_CUDA::load_kernels(// features
				      const std::size_t   Num_of_features_in,
				      const std::size_t   Num_of_features_out,
				      // weights
				      const int           Number_of_weights,
				      double**            Shared_weights,
				      double*             Shared_biases,
				      // Weights position and transposed matrix
				      std::size_t         Im_size_in,
				      std::size_t         Im_size_out,
				      std::size_t**       Weights_pos_oi,
				      std::size_t**       Weights_pos_io,
				      // ToDo: to remove
				      double* To_conv, double* Conv )
{
  //
  // Initialization
  std::cout << "Convolutional_CUDA -- Load kernels." << std::endl;
  // check on th device
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

  //
  // 1. Allocation space on the GPU
  // Features
  number_of_features_in_  = Num_of_features_in;
  number_of_features_out_ = Num_of_features_out;
  // weights
  number_of_weights_      = Number_of_weights;
  im_size_in_             = Im_size_in;
  im_size_out_            = Im_size_out;
  // Start allocation
  err = cudaMalloc((void **)&d_shared_weights_, Num_of_features_out * sizeof(double*) );
  err = cudaMalloc((void **)&d_shared_biases_,  Num_of_features_out * sizeof(double) );
  // Weights position and transposed matrix
  err = cudaMalloc((void **)&d_weights_pos_oi_, Im_size_out * Number_of_weights * sizeof(int) );
//  //err = cudaMalloc((void **)&d_weights_pos_io_, Im_size_in  * sizeof(double*) );
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  // weights
  for ( std::size_t p = 0 ; p < Num_of_features_out ; p++)
    {
      double *temp_weights;
      cudaMalloc((void **)&temp_weights, Number_of_weights * sizeof(double) );
      // create a master pointer we will move into the pointer to pointer
      cudaMemcpy(temp_weights, Shared_weights[p], Number_of_weights * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(&d_shared_weights_[p], &temp_weights, sizeof(double*), cudaMemcpyHostToDevice);
      if (err != cudaSuccess)
	{
	  fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
	  exit(EXIT_FAILURE);
	}
    }
  // Biases
  cudaMemcpy( d_shared_biases_, Shared_biases, Num_of_features_out * sizeof(double), cudaMemcpyHostToDevice );
  // Weights position
//  std::cout << "SIZE: " << Im_size_out * Number_of_weights * sizeof(std::size_t) << std::endl;
  std::size_t* weights_pos_oi = new std::size_t[Im_size_out * Number_of_weights];
  for ( int o = 0 ; o < Im_size_out ; o++ )
    for ( int k = 0 ; k < Number_of_weights ; k++ )
      {
	size_t idx          = k + o * Number_of_weights;
	//	std::cout << "Weights_pos_oi[o:"<< o << "][k:"<< k <<"] = " << Weights_pos_oi[o][k]<< std::endl;
	weights_pos_oi[idx] = Weights_pos_oi[o][k];
	//	std::cout << "weights_pos_oi[" << idx << "] = " << weights_pos_oi[idx] << std::endl;
      }
//  for (int k = 0 ; k < Number_of_weights ; k++)
//    std::cout << weights_pos_oi[k] << std::endl;
  err = cudaMemcpy( d_weights_pos_oi_, weights_pos_oi, Im_size_out * Number_of_weights * sizeof(int), cudaMemcpyHostToDevice );
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

//  for ( std::size_t p = 0 ; p < Im_size_out ; p++)
//    {
//      int *temp_out;
//      cudaMalloc((void **)&temp_out, Number_of_weights * sizeof(int) );
//      // create a master pointer we will move into the pointer to pointer
//      cudaMemcpy(temp_out, Weights_pos_oi[p], Number_of_weights * sizeof(int), cudaMemcpyHostToDevice);
//      cudaMemcpy(&d_weights_pos_oi_[p], &temp_out, sizeof(int*), cudaMemcpyHostToDevice);
//      if (err != cudaSuccess)
//	{
//	  fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
//	  exit(EXIT_FAILURE);
//	}
//    }

  //
  // test on conv
  double* d_to_conv;
  double* d_conv;
  //
  err = cudaMalloc((void **)&d_to_conv,   Im_size_in  * sizeof(double) );
  err = cudaMalloc((void **)&d_conv,      Im_size_out * sizeof(double) );
  //
  cudaMemcpy( d_to_conv, To_conv, Im_size_in  * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_conv,    Conv,    Im_size_out * sizeof(double), cudaMemcpyHostToDevice);
  //
  // 1. check on the device
  int threadsPerBlock = THREADSPERBLOCK;
  int Blocks_out      = (( Im_size_out ) + threadsPerBlock - 1) / threadsPerBlock;
  //
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  //
  std::cout << "Execute kernel" << std::endl;
  convolution_cuda<<< Blocks_out, threadsPerBlock >>>( static_cast< int >(im_size_in_),
						       static_cast< int >(im_size_out_),
						       number_of_weights_,
						       d_to_conv, d_conv,
						       d_shared_weights_, d_shared_biases_,
						       d_weights_pos_oi_ );
  //
  std::cout << "Copy back the data" << std::endl;
  cudaMemcpy( To_conv,
	      d_to_conv,
	      Im_size_out * sizeof(double), cudaMemcpyDeviceToHost );
  //
  cudaFree( d_to_conv );
  cudaFree( d_conv );
};
//
//
//
__host__ void
MAC::Convolutional_CUDA::forward()
{
};
//
//
//
__host__ void
MAC::Convolutional_CUDA::backward( std::map< std::string, Neurons_type >& Neurons,
				   const Functions& Activation_func )
{
};
//
//
//
__host__ 
MAC::Convolutional_CUDA::~Convolutional_CUDA()
{
  cudaError_t err = cudaGetLastError();
  //err = cudaFree( d_shared_weights_ );
  //err = cudaFree( d_shared_biases_ );
  //err = cudaFree( d_weights_pos_oi_ );
  //
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to free device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  // Reset the device and exit
  // cudaDeviceReset causes the driver to clean up all states. While
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
