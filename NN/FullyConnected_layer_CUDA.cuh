#ifndef FULLYCONNECTED_LAYER_CUDA_CUH
#define FULLYCONNECTED_LAYER_CUDA_CUH
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
//
//
//
#include "MACException.h"
//
//
//
namespace MAC
{

  /** \class FullyConnected_layer_CUDA
   *
   * \brief 
   * 
   * 
   */
  class FullyConnected_layer_CUDA
    {
    public:
      /** Constructor. */
      FullyConnected_layer_CUDA();
      //
      //explicit Subject( const int, const int );

      /** Destructor */
      virtual ~FullyConnected_layer_CUDA();

      //
      // Forward propagation
      __host__
      void forward(){};
      //
      //
      using  Neurons_type = std::tuple< std::vector< std::shared_ptr<double> > /* activations */,
	                                std::vector< std::shared_ptr<double> > /* neurons */,
	                                std::vector< std::shared_ptr<double> > /* deltas */  >;
      __host__
      void backward( std::map< std::string, Neurons_type >& );
      //
      //
      __host__
      void init( const int, const int*, const int, 
		 const double * );
      //
      //
      __host__
      void transpose_weight_matrices();

    private:
      //
      // CUDA layers initialized
      bool CUDA_init_{false};
      //
      // Weights and transposed weights
      double*   weights_{NULL};
      double*   E_{NULL};
      double* d_weights_{NULL};
      double* d_weights_T_{NULL};

      //
      //
      int  number_fc_layers_{0};
      int* fc_layers_{NULL};
      int  number_of_weights_{0};
      int  number_of_neurons_{0};
    };
}
#endif
