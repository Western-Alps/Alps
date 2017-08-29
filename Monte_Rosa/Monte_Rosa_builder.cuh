#ifndef MONTE_ROSA_BUILDER_CUH
#define MONTE_ROSA_BUILDER_CUH
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
#include "NeuralNetwork.cuh"
#include "NeuralNetworkComposite.cuh"
//
//
//
namespace MAC
{

  /** \class Monte_Rosa_builder
   *
   * \brief 
   * 
   * 
   */
  class Monte_Rosa_builder : public NeuralNetwork
    {
    public:
      /** Constructor. */
      Monte_Rosa_builder();
      //
      //explicit Subject( const int, const int );

      /** Destructor */
      virtual ~Monte_Rosa_builder(){};

      //
      // Forward propagation
      __host__
      virtual void forward(){};
      //
      //
      __host__ __device__
      virtual void backward(){};
      //
      //
      __host__
      virtual void add( std::shared_ptr< NeuralNetwork > ){};

    private:
      //
      //
      NeuralNetworkComposite mr_nn_;
    };
}
#endif
