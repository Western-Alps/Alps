#ifndef MONTE_ROSA_BUILDER_H
#define MONTE_ROSA_BUILDER_H
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
      // Initialization
      virtual void initialization();
      //
      // get the layer name
      virtual std::string get_layer_name(){ return std::string("Monte Rosa neural network.");};
      //
      // get the layer name
      virtual Layer get_layer_type(){ return Monte_rosa_layer;};
      //
      // Forward propagation
      virtual void forward( Subject&, const Weights& W = Weights() );
      //
      //
      virtual void backward();
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
