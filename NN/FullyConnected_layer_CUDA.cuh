#ifndef FULLYCONNECTED_LAYER_CUDA_CUH
#define FULLYCONNECTED_LAYER_CUDA_CUH
//
//
//
#include <iostream>
#include <fstream>
#include <math.h>  
#include <map>
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
      void forward();
      //
      //
      __host__ __device__
      void backward(){};
      //
      //
      __host__
      void add( ){};

    private:
      //
      // Weights
      double* weights_;
      double* d_weights_{NULL};
    };
}
#endif
