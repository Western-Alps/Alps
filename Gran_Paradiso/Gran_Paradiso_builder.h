#ifndef GRAN_PARADISO_BUILDER_H
#define GRAN_PARADISO_BUILDER_H
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
#include "Subject.h"
#include "NeuralNetwork.h"
#include "NeuralNetworkComposite.h"
//
//
//
namespace MAC
{

  /** \class Gran_Paradiso_builder
   *
   * \brief 
   * 
   * 
   */
  class Gran_Paradiso_builder : public NeuralNetwork
    {
    public:
      /** Constructor. */
      Gran_Paradiso_builder();
      //
      //explicit Subject( const int, const int );

      /** Destructor */
      virtual ~Gran_Paradiso_builder(){};

      //
      // Initialization
      virtual void initialization();
      //
      // get the layer name
      virtual std::string get_layer_name(){ return std::string("Monte Rosa neural network.");};
      //
      // get the layer name
      virtual Layer get_layer_type(){ return Gran_Paradiso_layer;};
      //
      // get the layer name
      virtual double get_energy(){ return 1. /*ToDo*/;};
      //
      // Forward propagation
      virtual void forward( Subject&, const Weights& W = Weights() );
      //
      //
      virtual void backward();
      //
      // Backward error propagation
      virtual void backward_error_propagation(){};
      //
      //
      virtual void add( std::shared_ptr< NeuralNetwork > ){};
      //
      //
      virtual int get_number_weights() const { return 1;};

    private:
      //
      //
      NeuralNetworkComposite mr_nn_;
    };
}
#endif
