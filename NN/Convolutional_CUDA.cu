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
test_cuda()
{
#if __CUDA_ARCH__ >= 200
  printf("Just a test");
#endif
}
//
//
//
__global__ void fill_with_zeros( int Image_size, double* Image )
{
  //
  //
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  //
  if ( idx <  Image_size )
    Image[idx] = 0.;
}
__global__ void cuda_hello()
{
  printf("Hello World from GPU!\n");

}
//
//
//
template< typename Activate >
__global__ void
convolution_cuda( int      Num_features_in,
		  int      Feature_out,
		  int      Image_size_out,
		  int      Number_of_weights,
		  double** To_conv,
		  double*  Conv,
		  double** Shared_weights,
		  double*  Shared_biases,
		  int*     Weights_pos_oi )
{
  //
  //
  int odx = blockDim.x * blockIdx.x + threadIdx.x;
  //
  if ( odx <  Image_size_out )
    {
      double   conv = 0.;
      Conv[odx]     = 0.;
      Activate activation;
      //
      for ( int feature = 0 ; feature < Num_features_in; feature++ )
	for ( int k = 0 ; k < Number_of_weights ; k++ )
	  {
	    int idx = k + odx * Number_of_weights;
	    conv += Shared_weights[Feature_out][k] * To_conv[feature][ Weights_pos_oi[idx] ];
	    //if ( Weights_pos_oi[idx] == 371232 )
	    //  printf("~ %d %d %d %f %d %f %f ~", odx, k, feature, Shared_weights[Feature_out][k], Weights_pos_oi[idx], To_conv[feature][ Weights_pos_oi[idx] ], conv );
	  }
      //
      Conv[odx] = /*activation.f(*/ conv + Shared_biases[Feature_out] /*)*/;
    }
}
//
//
//
template< typename Activate >
__global__ void
transpose_convolution_cuda( int      Num_features_in,
			    int      Feature_out,
			    int      Image_size_out,
			    int      Number_of_weights,
			    double** To_deconv,
			    double*  Deconv,
			    double** Shared_weights,
			    double*  Shared_biases,
			    int*     Weights_pos_io )
{
  //
  //
  int odx = blockDim.x * blockIdx.x + threadIdx.x;
  //
  if ( odx <  Image_size_out )
    {
      double deconv = 0.;
      Deconv[odx]   = 0.;
      Activate activation;
      //
      for ( int feature = 0 ; feature < Num_features_in; feature++ )
	for ( int k = 0 ; k < Number_of_weights ; k++ )
	  {
	    int idx = k + odx * Number_of_weights;
	    if ( Weights_pos_io[idx] != 999999999 )
	      deconv += Shared_weights[feature][k] * To_deconv[feature][ Weights_pos_io[idx] ];
	  }
      //
      Deconv[odx] = /*activation.f(*/ deconv + Shared_biases[Feature_out] /*)*/;
    }
}
//
//
//
__host__ 
MAC::Convolutional_CUDA::Convolutional_CUDA()
{}
//
//
//
__host__ void
MAC::Convolutional_CUDA::load_convolution_kernels(// features
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
						  std::size_t**       Weights_pos_io/*,
				      // ToDo: to remove
				      double* To_conv, double* Conv*/ )
{
  //
  // Initialization
  std::cout << "Convolutional_CUDA -- Load convolution kernels." << std::endl;
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
  // free the the device
  err = cudaDeviceReset();
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  // Start allocation
  err = cudaMalloc((void **)&d_shared_weights_, Num_of_features_out * sizeof(double*) );
  err = cudaMalloc((void **)&d_shared_biases_,  Num_of_features_out * sizeof(double) );
  // Weights position and transposed matrix
  err = cudaMalloc((void **)&d_weights_pos_oi_, Im_size_out * Number_of_weights * sizeof(int) );
  err = cudaMalloc((void **)&d_weights_pos_io_, Im_size_in  * Number_of_weights * sizeof(int) );
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
  int* weights_pos_oi = new int[Im_size_out * Number_of_weights];
  int* weights_pos_io = new int[Im_size_in  * Number_of_weights];
  //
  for ( std::size_t o = 0 ; o < Im_size_out ; o++ )
    for ( int k = 0 ; k < Number_of_weights ; k++ )
      {
	size_t odx          = k + o * Number_of_weights;
	weights_pos_oi[odx] = static_cast< int>( Weights_pos_oi[o][k] );
      }
  //
  for ( std::size_t i = 0 ; i < Im_size_in ; i++ )
    for ( int k = 0 ; k < Number_of_weights ; k++ )
      {
	size_t idx          = k + i * Number_of_weights;
	weights_pos_io[idx] = static_cast< int>( Weights_pos_io[i][k] );
      }
  //
  err = cudaMemcpy( d_weights_pos_oi_, weights_pos_oi, 
		    Im_size_out * Number_of_weights * sizeof(int), cudaMemcpyHostToDevice );
  err = cudaMemcpy( d_weights_pos_io_, weights_pos_io, 
		    Im_size_in  * Number_of_weights * sizeof(int), cudaMemcpyHostToDevice );
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
//toRm  //
//toRm  // test on conv
//toRm  double* d_to_conv;
//toRm  double* d_conv;
//toRm  //
//toRm  err = cudaMalloc((void **)&d_to_conv,   Im_size_in  * sizeof(double) );
//toRm  err = cudaMalloc((void **)&d_conv,      Im_size_out * sizeof(double) );
//toRm  //
//toRm  cudaMemcpy( d_to_conv, To_conv, Im_size_in  * sizeof(double), cudaMemcpyHostToDevice);
//toRm  cudaMemcpy( d_conv,    Conv,    Im_size_out * sizeof(double), cudaMemcpyHostToDevice);
//toRm  //
//toRm  // 1. check on the device
//toRm  int threadsPerBlock = THREADSPERBLOCK;
//toRm  int Blocks_out      = (( Im_size_out ) + threadsPerBlock - 1) / threadsPerBlock;
//toRm  //
//toRm  if (err != cudaSuccess)
//toRm    {
//toRm      fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
//toRm      exit(EXIT_FAILURE);
//toRm    }
//toRm  //
//toRm  std::cout << "Execute kernel" << std::endl;
//toRm  convolution_cuda<<< Blocks_out, threadsPerBlock >>>( static_cast< int >(im_size_in_),
//toRm						       static_cast< int >(im_size_out_),
//toRm						       number_of_weights_,
//toRm						       d_to_conv, d_conv,
//toRm						       d_shared_weights_, d_shared_biases_,
//toRm						       d_weights_pos_oi_ );
//toRm   //
//toRm  std::cout << "Copy back the data" << std::endl;
//toRm  cudaMemcpy( Conv, d_conv,
//toRm	      Im_size_out * sizeof(double), cudaMemcpyDeviceToHost );
//toRm  //
//toRm  cudaFree( d_to_conv );
//toRm  cudaFree( d_conv );
};
//
//
//
__host__ void
MAC::Convolutional_CUDA::load_deconvolution_kernels(// features
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
						    std::size_t**       Weights_pos_io )
{
  //
  // Initialization
  std::cout << "Convolutional_CUDA -- Load deconvolution kernels." << std::endl;
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
  // free the the device
  err = cudaDeviceReset();
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  // Start allocation
  err = cudaMalloc((void **)&d_shared_weights_, Num_of_features_in * sizeof(double*) );
  err = cudaMalloc((void **)&d_shared_biases_,  Num_of_features_out * sizeof(double) );
  // Weights position and transposed matrix
  err = cudaMalloc((void **)&d_weights_pos_oi_, Im_size_out * Number_of_weights * sizeof(int) );
  err = cudaMalloc((void **)&d_weights_pos_io_, Im_size_in  * Number_of_weights * sizeof(int) );
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  // weights
  for ( std::size_t p = 0 ; p < Num_of_features_in ; p++)
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
  int* weights_pos_oi = new int[Im_size_out * Number_of_weights];
  int* weights_pos_io = new int[Im_size_in  * Number_of_weights];
  //
  for ( std::size_t o = 0 ; o < Im_size_out ; o++ )
    for ( int k = 0 ; k < Number_of_weights ; k++ )
      {
	size_t odx          = k + o * Number_of_weights;
	weights_pos_oi[odx] = static_cast< int>( Weights_pos_oi[o][k] );
      }
  //
  for ( std::size_t i = 0 ; i < Im_size_in ; i++ )
    for ( int k = 0 ; k < Number_of_weights ; k++ )
      {
	size_t idx          = k + i * Number_of_weights;
	weights_pos_io[idx] = static_cast< int>( Weights_pos_io[i][k] );
      }
  //
  err = cudaMemcpy( d_weights_pos_oi_, weights_pos_oi, 
		    Im_size_out * Number_of_weights * sizeof(int), cudaMemcpyHostToDevice );
  err = cudaMemcpy( d_weights_pos_io_, weights_pos_io, 
		    Im_size_in  * Number_of_weights * sizeof(int), cudaMemcpyHostToDevice );
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
};
//
//
//
__host__ void
MAC::Convolutional_CUDA::load_feature_maps( double** Prev_feature_maps )
{
  //
  // Initialization
  std::cout << "Convolutional_CUDA -- Load previouse feature maps & prepare next maps." << std::endl;
  // check on th device
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

  //
  // 1. Allocation space for the previous features on the GPU
  err = cudaMalloc( (void **)&d_previouse_feature_maps_,
		    number_of_features_in_ * sizeof(double*) );

  for ( std::size_t p = 0 ; p < number_of_features_in_ ; p++)
    {
      double *temp_feature;
      cudaMalloc((void **)&temp_feature, im_size_in_ * sizeof(double) );
      // create a master pointer we will move into the pointer to pointer
      cudaMemcpy( temp_feature, Prev_feature_maps[p],
		  im_size_in_ * sizeof(double), cudaMemcpyHostToDevice );
      //
      cudaMemcpy(&d_previouse_feature_maps_[p], &temp_feature,
		 sizeof(double*), cudaMemcpyHostToDevice);
      if (err != cudaSuccess)
	{
	  fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
	  exit(EXIT_FAILURE);
	}
    }

  //
  // 2. Allocation space for the next features on the GPU
  err = cudaMalloc( (void **)&d_next_feature_maps_, im_size_out_ * sizeof(double) );

  //
  // 3. Fill the next features with zeros
  int threadsPerBlock = THREADSPERBLOCK;
  int Blocks_out      = (( im_size_out_ ) + threadsPerBlock - 1) / threadsPerBlock;
  fill_with_zeros<<< Blocks_out, threadsPerBlock >>>( im_size_out_,
						      d_next_feature_maps_ );
}
//
//
//
__host__ void
MAC::Convolutional_CUDA::convolution( double**         Next_feature_maps,
				      const Functions& Activation_func )
{
  std::cout << "Convolutional_CUDA -- Run the convolution." << std::endl;
  //
  // 1. check on the device and load the 
  int threadsPerBlock = THREADSPERBLOCK;
  int numBlocks       = (( im_size_out_ ) + threadsPerBlock - 1) / threadsPerBlock;
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
  //
  // 2. Convolution and move the maps
  switch( Activation_func.get_function_name() )
    {
    case Func::F_TANH:
      {
	for ( std::size_t feature = 0 ; feature < number_of_features_out_; feature++ )
	  {
	    // 2.1. convolution
	    convolution_cuda< MAC::Activation_tanh ><<< numBlocks, threadsPerBlock >>>
	      ( static_cast< int >(number_of_features_in_),
		static_cast< int >(feature),
		static_cast< int >(im_size_out_),
		number_of_weights_,
		d_previouse_feature_maps_,
		d_next_feature_maps_,
		d_shared_weights_, d_shared_biases_,
		d_weights_pos_oi_ );
	    // 2.2 move the feature map back
	    cudaMemcpy( Next_feature_maps[feature], d_next_feature_maps_,
			im_size_out_ * sizeof(double), cudaMemcpyDeviceToHost );
	    // 2.3 reset the feature map
	    fill_with_zeros<<< numBlocks, threadsPerBlock >>>
	      ( im_size_out_, d_next_feature_maps_ );
	  }
	break;
      }
    case Func::F_SIGMOID:
      {
	for ( std::size_t feature = 0 ; feature < number_of_features_out_; feature++ )
	  {
	    // 2.1. convolution
	    convolution_cuda< MAC::Activation_sigmoid ><<< numBlocks, threadsPerBlock >>>
	      ( static_cast< int >(number_of_features_in_),
		static_cast< int >(feature),
		static_cast< int >(im_size_out_),
		number_of_weights_,
		d_previouse_feature_maps_,
		d_next_feature_maps_,
		d_shared_weights_, d_shared_biases_,
		d_weights_pos_oi_ );
	    // 2.2 move the feature map back
	    cudaMemcpy( Next_feature_maps[feature], d_next_feature_maps_,
			im_size_out_ * sizeof(double), cudaMemcpyDeviceToHost );
	    // 2.3 reset the feature map
	    fill_with_zeros<<< numBlocks, threadsPerBlock >>>
	      ( im_size_out_, d_next_feature_maps_ );
	  }
	break;
      }
    case Func::UNDETERMINED:
    default:
      {
	fprintf(stderr, "Wrong activation function.\n");
	exit(EXIT_FAILURE);
      }
    }
}
//
//
//
__host__ void
MAC::Convolutional_CUDA::transpose_convolution( double**         Next_feature_maps,
						const Functions& Activation_func )
{
  std::cout << "Convolutional_CUDA -- Run the deconvolution." << std::endl;
  //
  // 1. check on the device and load the 
  int threadsPerBlock = THREADSPERBLOCK;
  int numBlocks       = (( im_size_out_ ) + threadsPerBlock - 1) / threadsPerBlock;
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
  //
  // 2. Convolution and move the maps
  switch( Activation_func.get_function_name() )
    {
    case Func::F_TANH:
      {
	for ( std::size_t feature = 0 ; feature < number_of_features_out_; feature++ )
	  {
	    // 2.1. convolution
	    transpose_convolution_cuda< MAC::Activation_tanh ><<< numBlocks, threadsPerBlock >>>
	      ( static_cast< int >(number_of_features_in_),
		static_cast< int >(feature),
		static_cast< int >(im_size_out_),
		number_of_weights_,
		d_previouse_feature_maps_,
		d_next_feature_maps_,
		d_shared_weights_, d_shared_biases_,
		d_weights_pos_oi_ );
	    // 2.2 move the feature map back
	    cudaMemcpy( Next_feature_maps[feature], d_next_feature_maps_,
			im_size_out_ * sizeof(double), cudaMemcpyDeviceToHost );
	    // 2.3 reset the feature map
	    fill_with_zeros<<< numBlocks, threadsPerBlock >>>
	      ( im_size_out_, d_next_feature_maps_ );
	  }
	break;
      }
    case Func::F_SIGMOID:
      {
	for ( std::size_t feature = 0 ; feature < number_of_features_out_; feature++ )
	  {
	    // 2.1. convolution
	    transpose_convolution_cuda< MAC::Activation_sigmoid ><<< numBlocks, threadsPerBlock >>>
	      ( static_cast< int >(number_of_features_in_),
		static_cast< int >(feature),
		static_cast< int >(im_size_out_),
		number_of_weights_,
		d_previouse_feature_maps_,
		d_next_feature_maps_,
		d_shared_weights_, d_shared_biases_,
		d_weights_pos_oi_ );
	    // 2.2 move the feature map back
	    cudaMemcpy( Next_feature_maps[feature], d_next_feature_maps_,
			im_size_out_ * sizeof(double), cudaMemcpyDeviceToHost );
	    // 2.3 reset the feature map
	    fill_with_zeros<<< numBlocks, threadsPerBlock >>>
	      ( im_size_out_, d_next_feature_maps_ );
	  }
	break;
      }
    case Func::UNDETERMINED:
    default:
      {
	fprintf(stderr, "Wrong activation function.\n");
	exit(EXIT_FAILURE);
      }
    }
}
//
//
//
__host__ void
MAC::Convolutional_CUDA::forward()
{
  std::cout << "Go fwd CUDA" << std::endl;
};
//
//
//
__host__ void
MAC::Convolutional_CUDA::backward( std::map< std::string, Neurons_type >& Neurons,
				   const Functions& Activation_func )
{
  std::cout << "Go bckwd CUDA" << std::endl;
};
//
//
//
__host__ 
MAC::Convolutional_CUDA::~Convolutional_CUDA()
{
  cudaError_t err = cudaGetLastError();
  err = cudaFree( d_shared_weights_ );
  err = cudaFree( d_shared_biases_ );
  err = cudaFree( d_weights_pos_oi_ );
  err = cudaFree( d_weights_pos_io_ );
  err = cudaFree( d_previouse_feature_maps_ );
  err = cudaFree( d_next_feature_maps_ );
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
  //
  if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
}
