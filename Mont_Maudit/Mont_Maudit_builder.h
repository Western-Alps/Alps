#ifndef MONT_MAUDIT_BUILDER_H
#define MONT_MAUDIT_BUILDER_H
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

  /** \class Mont_Maudit_builder
   *
   * \brief 
   * 
   * 
   */
  class Mont_Maudit_builder : public NeuralNetwork
    {
    public:
      /** Constructor. */
      Mont_Maudit_builder();
      //
      //explicit Subject( const int, const int );

      /** Destructor */
      virtual ~Mont_Maudit_builder(){};

      //
      // Initialization
      virtual void initialization();
      //
      // get the layer name
      virtual std::string get_layer_name(){ return std::string("Monte Rosa neural network.");};
      //
      // get the layer name
      virtual Layer get_layer_type(){ return Mont_Maudit_layer;};
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
