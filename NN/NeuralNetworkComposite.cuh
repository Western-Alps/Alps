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
#include "NN_test.cuh"
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
      NeuralNetworkComposite();
      //
      //explicit Subject( const int, const int );

      /** Destructor */
      virtual ~NeuralNetworkComposite();

      //
      // Forward propagation
      __host__
      virtual void forward()
      {
	for ( auto nn_elem : nn_composite_ )
	  {
	    std::cout << "New elem" << std::endl;
	    nn_elem->forward();
	  }
      };
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
      // Structure of the composite neural network
      std::list< std::shared_ptr< NeuralNetwork > > nn_composite_;

      //
      // Cuda error status
      cudaError_t cuda_err_{ cudaSuccess };

      
      //
      // Weights
      // weights on the host
      double* weights_;
      // weights ont the device
      double* d_weights_;
      
    };
}
#endif
