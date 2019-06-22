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
__host__ 
MAC::Convolutional_CUDA::Convolutional_CUDA()
{
}
//
//
//
__host__ void
MAC::Convolutional_CUDA::load_kernels( // features
				      const std::size_t   Num_of_features_in,
				      const std::size_t   Num_of_features_ou,
				      // weights
				      const int           Number_of_weights,
				      double**            Shared_weights,
				      double*             Shared_biases,
				      // Weights position and transposed matrix
				      std::size_t**       Weights_pos_oi,
				      std::size_t**       Weights_pos_io )
{
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
}
