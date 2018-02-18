#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
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
#include "MACLoadDataSet.h"
#include "Subject.h"
#include "Weights.h"
//
//
//
namespace MAC
{
  //
  // Enum on the type of layer
  enum Layer
  { /* Base*/
    neural_network_base_class,
    /* Leaves */
    neural_network_test_class,
    convolutional_layer,
    fully_connected_layer,
    convolutional_encoder,
    convolutional_decoder,
    /* Composite */
    composite_layer,
    /* Composition neural network */
    Monte_Rosa_layer,
    Mont_Blanc_layer,
    Mont_Maudit_layer
  };
 

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
      // Initialization
      virtual void initialization(){};
      //
      // get the layer name
      virtual std::string get_layer_name(){ return std::string("Neural network base class.");};
      //
      // get the layer name
      virtual Layer get_layer_type(){ return neural_network_base_class;};
      //
      // Forward propagation
      virtual void forward( Subject&, const Weights& W = Weights() ){};
      //
      // Backward propagation
      virtual void backward(){};
      //
      // Backward error propagation
      virtual void backward_error_propagation(){};
      //
      //
      virtual void add( std::shared_ptr< NeuralNetwork > ){};
      //
      //
      virtual int get_number_weights() const { return 1;};
    };
}
#endif
