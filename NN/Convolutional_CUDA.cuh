#ifndef CONVOLUTIONAL_CUDA_CUH
#define CONVOLUTIONAL_CUDA_CUH
//
//
//
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <math.h>  
#include <map>
#include <vector>
#include <list>
#include <memory>
//
// CUDA
//
#include <cuda_runtime.h>
// http://docs.nvidia.com/cuda/thrust/index.html
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
//
//
//
#include "MACException.h"
#include "Functions.h"
//
//
//
namespace MAC
{
  /** \class Convolutional_CUDA
   *
   * \brief 
   * 
   * 
   */
  class Convolutional_CUDA
    {
    public:
      //
      //
      using  Neurons_type = std::tuple< std::vector< std::shared_ptr<double> > /* activations */,
	                                std::vector< std::shared_ptr<double> > /* neurons */,
	                                std::vector< std::shared_ptr<double> > /* deltas */  >;

      /** Constructor. */
      Convolutional_CUDA();
      //
      //explicit Subject( const int, const int );

      /** Destructor */
      virtual ~Convolutional_CUDA();

      //
      // Initialization
      __host__
      void load_kernels( const std::size_t, const std::size_t,
			 const int,
			 double**, double*,
			 std::size_t,  std::size_t,
			 std::size_t**,  std::size_t**,
			 double*, double* );
      
      //
      // Forward/Backward propagation
      __host__
      void forward();
      __host__
      void backward( std::map< std::string, Neurons_type >&, const Functions&  );
      

    private:
      //
      // GPU (device) members
      // 

      //
      // features
      std::size_t     number_of_features_in_{0};
      std::size_t     number_of_features_out_{0};
      // weights
      int             number_of_weights_{0};
      double**      d_shared_weights_{NULL};
      double*       d_shared_biases_{NULL};
      // Weights position and transposed matrix
      std::size_t     im_size_in_{0};
      std::size_t     im_size_out_{0};
      int*  d_weights_pos_oi_{NULL};
      std::size_t** d_weights_pos_io_{NULL};
    };
}
#endif
