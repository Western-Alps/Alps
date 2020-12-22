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
//#include "AlpsLoadDataSet.h"
#include "AlpsClimber.h"
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
  class NeuralNetwork
    {
    public:
      /** Destructor */
      virtual ~NeuralNetwork(){};

      //
      // Initialization
      // virtual       void        initialization()           = 0;
      //
      // get the layer name
      virtual const std::string get_layer_name()                          const = 0;
      //
      // get the layer type
      //virtual Layer get_layer_type() = 0;
      //
      // get the layer name
      virtual const double      get_energy()                              const = 0;
      //
      //
      virtual const int         get_number_weights()                      const = 0;
      //
      // Forward propagation
      virtual       void        forward( std::shared_ptr< Alps::Climber > )     = 0;
      //
      // Backward propagation
      virtual       void        backward()                                      = 0;
      //
      // Backward error propagation
      //virtual      void backward_error_propagation() = 0;
      //
      //
      virtual       void        add( std::shared_ptr< NeuralNetwork > )          = 0;
    };
}
#endif
