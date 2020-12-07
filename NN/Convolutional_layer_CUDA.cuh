#ifndef CONVOLUTIONAL_LAYER_CUDA_CUH
#define CONVOLUTIONAL_LAYER_CUDA_CUH
//
//
//
#include <iostream>
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
  typedef struct {
    int d_half_window_[4];
    int d_image_size_[3];
  } small_arrays;
  /** \class Convolutional_layer_CUDA
   *
   * \brief 
   * 
   * 
   */
  class Convolutional_layer_CUDA
    {
    public:
      /** Constructor. */
      Convolutional_layer_CUDA();
      //
      //explicit Subject( const int, const int );

      /** Destructor */
      virtual ~Convolutional_layer_CUDA();

      //
      // Forward propagation
      __host__
      void forward();
      //
      //
      using  Neurons_type = std::tuple< std::vector< std::shared_ptr<double> > /* activations */,
	                                std::vector< std::shared_ptr<double> > /* neurons */,
	                                std::vector< std::shared_ptr<double> > /* deltas */  >;
      __host__
      void backward( std::map< std::string, Neurons_type >&, const Functions&  );
      //
      //
      __host__
      void init( const int*, const int*, const int, const double* );

      //
      //
      __host__
      void transpose_weight_matrices();

      //
      //
      __host__
      void load_previouse_feature_maps( double**, Mapping*,
					const int, const int );

      //
      //
      __host__
      void load_previouse_feature_maps( double**, double**, Mapping*,
					const int, const int, const int );
      //
      //
      __host__
      void convolution( Neurons_type&,
			const int,
			const Functions&);
      //
      //
      __host__
      void convolution_decoding( Neurons_type&,
				 double&, 
				 const int,
				 const Functions& );
      //
      //
      __host__
      void set_weights_T( const int     Num_weights,
			  const double* Weights_T )
      {
	//
	//
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	  {
	    fprintf(stderr, "error on the CUDA device (error code %s)!\n", cudaGetErrorString(err));
	    exit(EXIT_FAILURE);
	  }
	//
	// Number of transposed weights
	number_of_weights_T_ = Num_weights;
	err = cudaMalloc ( (void **)&d_weights_T_, Num_weights * sizeof(double) );
	err = cudaMemcpy( d_weights_T_, Weights_T,  Num_weights * sizeof(double), 
			  cudaMemcpyHostToDevice );
      };

    private:
      //
      // CUDA layers initialized
      bool CUDA_init_{false};
      //
      // Weights and transposed weights
      //
      small_arrays to_cuda_;
      //int     d_half_window_{NULL};
      // int*    d_image_size_{NULL};
      double* d_weights_{NULL};
      double* d_weights_T_{NULL};
      double* d_E_{NULL};
      //
      double* d_activations_{NULL};
      double* d_neurons_{NULL};
      double* d_deltas_{NULL};

      //
      // Kernel functionalities
      // Feature maps, from the previouse layer
      Mapping* d_map_idx_;
      double** d_prev_feature_maps_{NULL};
      double** d_target_maps_{NULL};
      int      num_prev_images_{0};
      int      num_target_images_{0};


      //
      //
      int* image_size_{NULL};
      int* half_window_{NULL};
      int  number_of_weights_{0};
      int  number_of_weights_T_{0};
      int  number_of_neurons_{0};

      //
      // Gradient descent
      double learning_rate_{0.01};
    };
}
#endif
