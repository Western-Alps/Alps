#ifndef ALPSNEURALNETWORK_H
#define ALPSNEURALNETWORK_H
//
//
//
#include <iostream>
#include <fstream>
#include <math.h>  
#include <map>
#include <memory>
#include <list>
//
// CUDA
//
#include <cuda_runtime.h>
//
//
//
#include "MACException.h"
#include "AlpsLayer.h"
#include "Weights.h"
//
//
//
namespace Alps
{
  /** \class AlpsNeuralNetwork
   *
   * \brief 
   * This class is the head of a composit design to build neural network
   * 
   */
  class NeuralNetwork : public Alps::Layer
    {
    public:
      /** Destructor */
      virtual ~NeuralNetwork(){};


      //
      // Accessors
      //
      // get neural network energy
      virtual const double get_energy()                          const = 0;


      //
      // Functions
      //
      // Add a layer in the neural network
      virtual       void   add( std::shared_ptr< Alps::Layer > )       = 0;
    };
}
#endif
