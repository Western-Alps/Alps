//
//
//
#include "MACException.h"
#include "Convolutional_layer_CUDA.cuh"
#include "Activations.h"
//
#define THREADSPERBLOCK 1024
//
// Note that any atomic operation can be implemented based on atomicCAS() (Compare And Swap). For example, atomicAdd() for double-precision floating-point numbers is not available on devices with compute capability lower than 6.0 but it can be implemented as follows: 
#if __CUDA_ARCH__ < 600
__device__ double atomicAdd_patch(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif
//
//
__global__ void
test_cuda( double **tempo, MAC::Mapping* Map, int mod )
{
  printf("image(%d) = %f \n", mod, tempo[mod][421578]);
  printf("Map(%d) = [%d,%d,%d] \n", 421578, Map[421578].x_, Map[421578].y_, Map[421578].z_);
}
/**
 * CUDA Kernel Device code
 *
 * Computes the 3D convolution
 */
template< typename Activate >
__global__ void
 convolution_cuda( double** Previouse_feature_maps, MAC::Mapping* Map_idx,
		   double*  Activations, double*  Neurons, double*  Weights,
		   const MAC::small_arrays To_cuda /*int* Half_window, int* Image_size*/,
		   const int Number_of_weights, const int Number_of_neurons,
		   const int Num_prev_images,
		   const int Modality )
{
  //
  // We visit all voxels
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  // if we are inside of the image
  if ( idx < Number_of_neurons )
    {
      //
      // Activation function
      Activate a;
      //
      int
	Window_x = 2*To_cuda.d_half_window_[1]+1,
	Window_y = 2*To_cuda.d_half_window_[2]+1,
	Window_z = 2*To_cuda.d_half_window_[3]+1;
      //
      int
	vox_x = Map_idx[idx].x_,
	vox_y = Map_idx[idx].y_,
	vox_z = Map_idx[idx].z_;
      //
      // Convolution
      double convolution_voxel_value = 0.;
      if ( vox_x - To_cuda.d_half_window_[1] > -1 && vox_x + To_cuda.d_half_window_[1] < To_cuda.d_image_size_[0] &&
	   vox_y - To_cuda.d_half_window_[2] > -1 && vox_y + To_cuda.d_half_window_[2] < To_cuda.d_image_size_[1] &&
	   vox_z - To_cuda.d_half_window_[3] > -1 && vox_z + To_cuda.d_half_window_[3] < To_cuda.d_image_size_[2]  )
	{
	  for ( int prev = 0 ; prev < Num_prev_images ; prev++ ) 
	    for ( int z = -To_cuda.d_half_window_[3]; z < To_cuda.d_half_window_[3]+1 ; z++ ) // run through z
	      for( int y = -To_cuda.d_half_window_[2]; y < To_cuda.d_half_window_[2]+1 ; y++ ) // run through y
		for( int x = -To_cuda.d_half_window_[1]; x < To_cuda.d_half_window_[1]+1 ; x++ ) // run through x
		    {
		      int weight_idx = (x+To_cuda.d_half_window_[1])
			+ Window_x * (y+To_cuda.d_half_window_[2])
			+ Window_x * Window_y
			* (z+To_cuda.d_half_window_[3])
			+ Window_x * Window_y
			* Window_z * Modality;
		      //
		      int neighbor = (vox_x+x) + To_cuda.d_image_size_[0] * (vox_y+y)
			+ To_cuda.d_image_size_[0]*To_cuda.d_image_size_[1]*(vox_z+z);
		      convolution_voxel_value += Weights[ weight_idx ] * Previouse_feature_maps[prev][neighbor];
		    }
	  // add the bias at the end of the array
	  int bias_position =  To_cuda.d_half_window_[0] * Window_x * Window_y * Window_z + Modality;
	  convolution_voxel_value += Weights[ bias_position ]; // x 1.
	}
      else
	{
	  for ( int prev = 0 ; prev < Num_prev_images ; prev++ ) 
	    for ( int z = -To_cuda.d_half_window_[3]; z < To_cuda.d_half_window_[3]+1 ; z++ ) // run through z
	      for( int y = -To_cuda.d_half_window_[2]; y < To_cuda.d_half_window_[2]+1 ; y++ ) // run through y
		for( int x = -To_cuda.d_half_window_[1]; x < To_cuda.d_half_window_[1]+1 ; x++ ) // run through x
		  if( vox_x + x > -1 && vox_y + y > -1 && vox_z + z > -1 &&
		      vox_x + x < static_cast<int>(To_cuda.d_image_size_[0]) &&
		      vox_y + y < static_cast<int>(To_cuda.d_image_size_[1]) &&
		      vox_z + z < static_cast<int>(To_cuda.d_image_size_[2]) ) // zero padding
		    {
		      int weight_idx = (x+To_cuda.d_half_window_[1])
			+ Window_x * (y+To_cuda.d_half_window_[2])
			+ Window_x * Window_y
			* (z+To_cuda.d_half_window_[3])
			+ Window_x * Window_y
			* Window_z * Modality;
		      //
		      int neighbor = (vox_x+x) + To_cuda.d_image_size_[0] * (vox_y+y)
			+ To_cuda.d_image_size_[0]*To_cuda.d_image_size_[1]*(vox_z+z);
		      convolution_voxel_value += Weights[ weight_idx ] * Previouse_feature_maps[prev][neighbor];
		    }
	  // add the bias at the end of the array
	  int bias_position = To_cuda.d_half_window_[0] * Window_x * Window_y * Window_z + Modality;
	  convolution_voxel_value += Weights[ bias_position ]; // x 1.
	}
      //
      // record the activation
      Activations[idx] = convolution_voxel_value;
      Neurons[idx]     = a.f(convolution_voxel_value);
//      if ( idx == 421578 )
//	{
//	  //printf("(%d,%d,%d) = %f = [%d,%d,%d]  ", x,y,z,
//	  //	 Previouse_feature_maps[prev][neighbor],
//	  //	 (vox_x+x),(vox_y+y),(vox_z+z));
//	  // printf(" Weights[ %d ] = %f ", weight_idx, Weights[ weight_idx ] );
//	  printf(" (%f, %f) ", convolution_voxel_value, a.f(convolution_voxel_value) );
//	}
    }
}
/**
 * CUDA Kernel Device code
 *
 * Computes the 3D convolution
 */
template< typename Activate >
__global__ void
 convolution_decoding_cuda( double** Previouse_feature_maps, double** Target_maps, MAC::Mapping* Map_idx,
			    double*  Activations, double*  Neurons, double*  Deltas, double*  Weights_T,
			    const MAC::small_arrays To_cuda /*int* Half_window, int* Image_size*/,
			    const int Number_of_weights, const int Number_of_neurons,
			    const int Num_prev_images,
			    const int Modality )
{
  //
  // We visit all voxels
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  // if we are inside of the image
  if ( idx < Number_of_neurons )
    {
      //
      // Activation function
      Activate a;
      //
      int
	Window_x = 2*To_cuda.d_half_window_[1]+1,
	Window_y = 2*To_cuda.d_half_window_[2]+1,
	Window_z = 2*To_cuda.d_half_window_[3]+1;
      //
      int
	vox_x = Map_idx[idx].x_,
	vox_y = Map_idx[idx].y_,
	vox_z = Map_idx[idx].z_;
      //
      // Convolution
      double convolution_voxel_value = 0.;
      if ( vox_x - To_cuda.d_half_window_[1] > -1 && vox_x + To_cuda.d_half_window_[1] < To_cuda.d_image_size_[0] &&
	   vox_y - To_cuda.d_half_window_[2] > -1 && vox_y + To_cuda.d_half_window_[2] < To_cuda.d_image_size_[1] &&
	   vox_z - To_cuda.d_half_window_[3] > -1 && vox_z + To_cuda.d_half_window_[3] < To_cuda.d_image_size_[2]  )
	{
	  for ( int prev = 0 ; prev < Num_prev_images ; prev++ ) 
	    for ( int z = -To_cuda.d_half_window_[3]; z < To_cuda.d_half_window_[3]+1 ; z++ ) // run through z
	      for( int y = -To_cuda.d_half_window_[2]; y < To_cuda.d_half_window_[2]+1 ; y++ ) // run through y
		for( int x = -To_cuda.d_half_window_[1]; x < To_cuda.d_half_window_[1]+1 ; x++ ) // run through x
		    {
		      int weight_idx = (x+To_cuda.d_half_window_[1])
			+ Window_x * (y+To_cuda.d_half_window_[2])
			+ Window_x * Window_y
			* (z+To_cuda.d_half_window_[3])
			+ Window_x * Window_y
			* Window_z * prev;
		      //
		      int neighbor = (vox_x+x) + To_cuda.d_image_size_[0] * (vox_y+y)
			+ To_cuda.d_image_size_[0]*To_cuda.d_image_size_[1]*(vox_z+z);
		      convolution_voxel_value += Weights_T[ weight_idx ] * Previouse_feature_maps[prev][neighbor];
		    }
	  // add the bias at the end of the array
	  int bias_position =  To_cuda.d_half_window_[0] * Window_x * Window_y * Window_z + Modality;
	  convolution_voxel_value += Weights_T[ bias_position ]; // x 1.
	}
      else
	{
	  for ( int prev = 0 ; prev < Num_prev_images ; prev++ ) 
	    for ( int z = -To_cuda.d_half_window_[3]; z < To_cuda.d_half_window_[3]+1 ; z++ ) // run through z
	      for( int y = -To_cuda.d_half_window_[2]; y < To_cuda.d_half_window_[2]+1 ; y++ ) // run through y
		for( int x = -To_cuda.d_half_window_[1]; x < To_cuda.d_half_window_[1]+1 ; x++ ) // run through x
		  if( vox_x + x > -1 && vox_y + y > -1 && vox_z + z > -1 &&
		      vox_x + x < static_cast<int>(To_cuda.d_image_size_[0]) &&
		      vox_y + y < static_cast<int>(To_cuda.d_image_size_[1]) &&
		      vox_z + z < static_cast<int>(To_cuda.d_image_size_[2]) ) // zero padding
		    {
		      int weight_idx = (x+To_cuda.d_half_window_[1])
			+ Window_x * (y+To_cuda.d_half_window_[2])
			+ Window_x * Window_y
			* (z+To_cuda.d_half_window_[3])
			+ Window_x * Window_y
			* Window_z * prev;
		      //
		      int neighbor = (vox_x+x) + To_cuda.d_image_size_[0] * (vox_y+y)
			+ To_cuda.d_image_size_[0]*To_cuda.d_image_size_[1]*(vox_z+z);
		      convolution_voxel_value += Weights_T[ weight_idx ] * Previouse_feature_maps[prev][neighbor];
		    }
	  // add the bias at the end of the array
	  int bias_position = To_cuda.d_half_window_[0] * Window_x * Window_y * Window_z + Modality;
	  convolution_voxel_value += Weights_T[ bias_position ]; // x 1.
	}
      //
      // record the activation
      Activations[idx] = convolution_voxel_value;
      //
      double function_activation = a.f(convolution_voxel_value);
      double delta               = Target_maps[Modality][idx] - function_activation;
      Neurons[idx]               = function_activation;
      Deltas[idx]                = delta;
      //
      //if ( idx == 421578 )
      //	{
      //	  printf(" [%f, %f, %f] ", convolution_voxel_value, Target_maps[Modality][idx], delta );
      //	  printf("Map(%d) = [%d,%d,%d] \n", 421578, Map_idx[421578].x_, Map_idx[421578].y_, Map_idx[421578].z_);
      //	}
      //if ( idx == 411578 )
      //	{
      //	  printf(" [%f, %f, %f] ", convolution_voxel_value, Target_maps[Modality][idx], delta );
      //	  printf("Map(%d) = [%d,%d,%d] \n", 411578, Map_idx[411578].x_, Map_idx[411578].y_, Map_idx[411578].z_);
      //	}
    }
}
/**
 * CUDA Kernel Device code
 *
 * Sum of squares to calculate the energy
 */
__global__ void
sum_of_squares( double* Deltas, double* E_i, const int Number_of_neurons ) {
  //
  // Calculation of energy per block
  __shared__ double  block_energy[ THREADSPERBLOCK ];
   
  //
  // We visit all voxels
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  // if we are inside of the image
  if ( idx < Number_of_neurons )
    {
      double delta = Deltas[ idx ];
      //printf(" %f ", delta);
      block_energy[ threadIdx.x ] = delta * delta;
    }
  else
    block_energy[ threadIdx.x ] = 0.;
  // thread synchronization
  __syncthreads();

  //
  //
  if( 0 == threadIdx.x )
    {
      double sum = 0;
      for( int i = 0 ; i < THREADSPERBLOCK ; i++ )
	sum += block_energy[i];
      atomicAdd_patch( E_i, sum );
      //printf(" %f ", *E_i);
    }
}
//
//
//
MAC::Convolutional_layer_CUDA::Convolutional_layer_CUDA()
{};
//
//
//
__host__ void
MAC::Convolutional_layer_CUDA::init( const int*    Image_size,
				     const int*    Half_window,
				     const int     Number_of_weights, 
				     const double* Weights )
{
  if ( !CUDA_init_ )
    {
      //
      // Initialization
      std::cout << "CUDA Convolutional_layer_CUDA init." << std::endl;
      //
      image_size_ = new int[3];
      memcpy( image_size_, Image_size, 3*sizeof(int) );
      //
      half_window_ = new int[4];
      memcpy( half_window_, Half_window, 4*sizeof(int) );
      //
      number_of_weights_ = Number_of_weights;
      // Compute number of neurons
      number_of_neurons_ += image_size_[0]*image_size_[1]*image_size_[2];

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
//      err = cudaMalloc((void **)&d_E_,         number_of_weights_* sizeof(double) );
      //1err = cudaMalloc((void **)&d_half_window_, 4 * sizeof(int) );
      //1err = cudaMalloc((void **)&d_image_size_,  3 * sizeof(int) );
      err = cudaMalloc((void **)&d_weights_, number_of_weights_ * sizeof(double) );
//      err = cudaMalloc((void **)&d_weights_T_, number_of_weights_* sizeof(double) );
      err = cudaMalloc((void **)&d_activations_, number_of_neurons_ * sizeof(double) );
      err = cudaMalloc((void **)&d_neurons_,     number_of_neurons_ * sizeof(double) );
      err = cudaMalloc((void **)&d_deltas_,      number_of_neurons_ * sizeof(double) );
//      // device
//      int threadsPerBlock = THREADSPERBLOCK;
//      int numBlocks = (( number_of_weights_ ) + threadsPerBlock - 1) / threadsPerBlock;
//      // 3.1. Compute delta
//      array_init_cuda<<< numBlocks, threadsPerBlock >>>( d_E_, number_of_weights_ );


      //
      // 2. Initialize on the device
      err = cudaMemset(d_activations_, 0., number_of_neurons_ * sizeof(double) );
      err = cudaMemset(d_neurons_,     0., number_of_neurons_ * sizeof(double) );
      err = cudaMemset(d_deltas_,      0., number_of_neurons_ * sizeof(double) );
      
      //
      // 3. Copy on the device
      //      memcpy( d_half_window_, half_window_, 4 * sizeof(int) );
      memcpy( to_cuda_.d_half_window_, half_window_, 4 * sizeof(int) );
      memcpy( to_cuda_.d_image_size_, image_size_, 3 * sizeof(int) );
//1      err = cudaMemcpy( d_half_window_, half_window_, 4 * sizeof(int), 
//1			cudaMemcpyHostToDevice );
//1      err = cudaMemcpy( d_image_size_,  image_size_,  3 * sizeof(int), 
//1			cudaMemcpyHostToDevice );
      err = cudaMemcpy( d_weights_, Weights, number_of_weights_ * sizeof(double), 
			cudaMemcpyHostToDevice );

      //
      // initialization done
      CUDA_init_ = true;
    }
}
//
//
//
__host__ void
MAC::Convolutional_layer_CUDA::load_previouse_feature_maps( double** Prev,
							    Mapping* Map,
							    const int Num_images,
							    const int Image_size )
{
  //
  // check on th device
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
  //
  // Allocation of the number of previouse images
  num_prev_images_ = Num_images;
  err = cudaMalloc((void **)&d_prev_feature_maps_, Num_images * sizeof(double*) );
  err = cudaMalloc((void **)&d_map_idx_, Image_size * sizeof(Mapping) );
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  // Allocate the index mapping
  cudaMemcpy(d_map_idx_, Map, Image_size * sizeof(Mapping), cudaMemcpyHostToDevice);
  // Allocation of the size of the previouse images
  for ( int p = 0 ; p < Num_images ; p++)
    {
      double* temp;
      cudaMalloc((void **)&temp, Image_size * sizeof(double) );
      // create a master pointer we will move into the pointer to pointer
      cudaMemcpy(temp, Prev[p],  Image_size * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(&d_prev_feature_maps_[p], &temp, sizeof(double*), cudaMemcpyHostToDevice);
      if (err != cudaSuccess)
	{
	  fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
	  exit(EXIT_FAILURE);
	}
      //cudaFree(temp);
      //test_cuda<<<1,1>>>( d_prev_feature_maps_, d_map_idx_, p );
    }
}
//
//
//
__host__ void
MAC::Convolutional_layer_CUDA::load_previouse_feature_maps( double** Prev,
							    double** Target,
							    Mapping* Map,
							    const int Num_images,
							    const int Num_targets,
							    const int Image_size )
{
  //
  // check on th device
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
  //
  // Allocation of the number of previouse images
  num_prev_images_   = Num_images;
  num_target_images_ = Num_targets;
  err = cudaMalloc((void **)&d_prev_feature_maps_, Num_images * sizeof(double*) );
  err = cudaMalloc((void **)&d_target_maps_, Num_targets * sizeof(double*) );
  err = cudaMalloc((void **)&d_map_idx_, Image_size * sizeof(Mapping) );
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  // Allocate the index mapping
  cudaMemcpy(d_map_idx_, Map, Image_size * sizeof(Mapping), cudaMemcpyHostToDevice);
  // Allocation of the size of the previouse images
  for ( int p = 0 ; p < Num_images ; p++)
    {
      double* temp;
      cudaMalloc((void **)&temp, Image_size * sizeof(double) );
      // create a master pointer we will move into the pointer to pointer
      cudaMemcpy(temp, Prev[p],  Image_size * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(&d_prev_feature_maps_[p], &temp, sizeof(double*), cudaMemcpyHostToDevice);
      if (err != cudaSuccess)
	{
	  fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
	  exit(EXIT_FAILURE);
	}
      //cudaFree(temp);
      //test_cuda<<<1,1>>>( d_prev_feature_maps_, d_map_idx_, p );
    }
  for ( int t = 0 ; t < Num_targets ; t++)
    {
      double* temp;
      cudaMalloc((void **)&temp, Image_size * sizeof(double) );
      // create a master pointer we will move into the pointer to pointer
      cudaMemcpy(temp, Target[t],  Image_size * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(&d_target_maps_[t], &temp, sizeof(double*), cudaMemcpyHostToDevice);
      if (err != cudaSuccess)
	{
	  fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
	  exit(EXIT_FAILURE);
	}
      //cudaFree(temp);
      //test_cuda<<<1,1>>>( d_target_maps_, d_map_idx_, t );
    }
}
//
//
//
__host__ void
MAC::Convolutional_layer_CUDA::convolution( Neurons_type& Sub, const int Mod,
					    const Functions& Activation_func )
{
  //
  // 1. check on the device
  int threadsPerBlock = THREADSPERBLOCK;
  int numBlocks       = (( number_of_neurons_ ) + threadsPerBlock - 1) / threadsPerBlock;
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

  //
  // 2. Convolution
  switch( Activation_func.get_function_name() )
    {
    case Func::F_TANH:
      convolution_cuda< MAC::Activation_tanh ><<< numBlocks, threadsPerBlock >>>( d_prev_feature_maps_,
										  d_map_idx_,
										  d_activations_,
										  d_neurons_,
										  d_weights_,
										  //d_half_window_,
										  to_cuda_,
										  //d_image_size_,
										  number_of_weights_,
										  number_of_neurons_,
										  num_prev_images_,
										  Mod );
      break;
    case Func::F_SIGMOID:
      convolution_cuda< MAC::Activation_sigmoid ><<< numBlocks, threadsPerBlock >>>( d_prev_feature_maps_,
										     d_map_idx_,
										     d_activations_,
										     d_neurons_,
										     d_weights_,
										     //d_half_window_,
										     to_cuda_,
										     //d_image_size_,
										     number_of_weights_,
										     number_of_neurons_,
										     num_prev_images_,
										     Mod );
      break;
    case Func::UNDETERMINED:
    default:
      {
	fprintf(stderr, "Wrong activation function.\n");
	exit(EXIT_FAILURE);
      }
    }

  //
  // 3. Download the the new feature map from GPU
  err = cudaMemcpy( (std::get< 0/*activations*/>( Sub ))[Mod].get(),
		    d_activations_,
		    number_of_neurons_ * sizeof(double), cudaMemcpyDeviceToHost );
  err = cudaMemcpy( (std::get< 1/*neurons*/>( Sub ))[Mod].get(),
		    d_neurons_,
		    number_of_neurons_ * sizeof(double), cudaMemcpyDeviceToHost );
}
//
//
//
__host__ void
MAC::Convolutional_layer_CUDA::convolution_decoding( Neurons_type& Sub, double& E_i, const int Mod,
						     const Functions& Activation_func )
{
  //
  // 1. check on the device
  int threadsPerBlock = THREADSPERBLOCK;
  int numBlocks       = (( number_of_neurons_ ) + threadsPerBlock - 1) / threadsPerBlock;
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

  //
  // 2. Convolution
  switch( Activation_func.get_function_name() )
    {
    case Func::F_TANH:
      convolution_decoding_cuda< MAC::Activation_tanh ><<< numBlocks, threadsPerBlock >>>( d_prev_feature_maps_,
											   d_target_maps_,
											   d_map_idx_,
											   d_activations_,
											   d_neurons_,
											   d_deltas_,
											   d_weights_T_,
											   //d_half_window_,
											   to_cuda_,
											   //d_image_size_,
											   number_of_weights_,
											   number_of_neurons_,
											   num_prev_images_,
											   Mod );
      break;
    case Func::F_SIGMOID:
      convolution_decoding_cuda< MAC::Activation_sigmoid ><<< numBlocks, threadsPerBlock >>>( d_prev_feature_maps_,
											      d_target_maps_,
											      d_map_idx_,
											      d_activations_,
											      d_neurons_,
											      d_deltas_,
											      d_weights_T_,
											      //d_half_window_,
											      to_cuda_,
											      //d_image_size_,
											      number_of_weights_,
											      number_of_neurons_,
											      num_prev_images_,
											      Mod );
      break;
    case Func::UNDETERMINED:
    default:
      {
	fprintf(stderr, "Wrong activation function.\n");
	exit(EXIT_FAILURE);
      }
    }

  //
  // 3. Download the the new feature map from GPU
  err = cudaMemcpy( (std::get< 0/*activations*/>( Sub ))[Mod].get(),
		    d_activations_,
		    number_of_neurons_ * sizeof(double), cudaMemcpyDeviceToHost );
  err = cudaMemcpy( (std::get< 1/*neurons*/>( Sub ))[Mod].get(),
		    d_neurons_,
		    number_of_neurons_ * sizeof(double), cudaMemcpyDeviceToHost );
  err = cudaMemcpy( (std::get< 2/*deltas*/>( Sub ))[Mod].get(),
		    d_deltas_,
		    number_of_neurons_ * sizeof(double), cudaMemcpyDeviceToHost );

  //
  // 4. Energy calculation
  double *d_energy, map_energy = 0.;
  cudaMalloc( (void**)& d_energy, sizeof(double) );
  cudaMemcpy( d_energy, &map_energy,  sizeof(double), cudaMemcpyHostToDevice);
  sum_of_squares<<< numBlocks, threadsPerBlock >>>( d_deltas_, d_energy, number_of_neurons_ );
  cudaMemcpy( &map_energy, d_energy, sizeof(double), cudaMemcpyDeviceToHost);
  //
  E_i += map_energy;

  //
  // Free
  cudaFree(d_energy);
}
//
//
//
__host__ void
MAC::Convolutional_layer_CUDA::transpose_weight_matrices()
{
}
//
//
//
__host__ void
MAC::Convolutional_layer_CUDA::forward(  )
{
};
//
//
//
__host__ void
MAC::Convolutional_layer_CUDA::backward( std::map< std::string, Neurons_type >& Neurons,
					  const Functions& Activation_func )
{
};
//
//
//
__host__ 
MAC::Convolutional_layer_CUDA::~Convolutional_layer_CUDA()
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
