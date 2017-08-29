#ifndef NEURALNETWORK_CUH
#define NEURALNETWORK_CUH
//
//
//
#include <iostream>
#include <fstream>
#include <math.h>  
#include <map>
#include <list>
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

  /** \class NeuralNetwork
   *
   * \brief 
   * This class is the head of a composit design to build neural network
   * 
   */
  class NeuralNetwork
    {
 
    protected:
      /** Constructor. */
      NeuralNetwork(){};
      //
      //explicit Subject( const int, const int );

    public:
      /** Destructor */
      virtual ~NeuralNetwork(){};

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
      virtual void add( NeuralNetwork* ){};
    };
}
#endif
