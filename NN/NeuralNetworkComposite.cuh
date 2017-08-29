#ifndef NEURALNETWORKCOMPOSITE_CUH
#define NEURALNETWORKCOMPOSITE_CUH
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
//
//
//
namespace MAC
{

  /** \class NeuralNetworkComposite
   *
   * \brief 
   * 
   * 
   */
  class NeuralNetworkComposite : public NeuralNetwork
    {
    public:
      /** Constructor. */
      NeuralNetworkComposite():
	NeuralNetwork::NeuralNetwork()
      {};
      //
      //explicit Subject( const int, const int );

      /** Destructor */
      virtual ~NeuralNetworkComposite(){};

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
      std::list< std::shared_ptr< NeuralNetwork > > nn_composite_;
    };
}
#endif
