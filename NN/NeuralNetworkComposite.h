#ifndef NEURALNETWORKCOMPOSITE_H
#define NEURALNETWORKCOMPOSITE_H
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
#include "NeuralNetwork.h"
#include "Subject.h"
#include "Weights.h"
#include "NN_test.h"
#include "Convolutional_layer.h"
#include "FullyConnected_layer.h"
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
      // Initialization
      virtual void initialization();
      //
      // get the layer name
      virtual std::string get_layer_name(){ return layer_name_;};
      //
      // get the layer name
      virtual Layer get_layer_type(){ return composite_layer;};
      //
      // Forward propagation
      virtual void forward( Subject& Sub, const Weights& W = Weights() )
      {
	for ( auto nn_elem : nn_composite_ )
	  {
	    std::cout << "New elem" << std::endl;
	    nn_elem->forward( Sub, weights_ );
	  }
      };
      //
      //
      virtual void backward(){};
      //
      //
      virtual void add( std::shared_ptr< NeuralNetwork > NN)
      {
	nn_composite_.push_back( NN );
      };
      //
      //
      virtual int get_number_weights() const { return number_of_weights_; };

    private:
      //
      // Convolutional layer's name
      std::string layer_name_{"composit layer"};
      // Type of layer
      Layer layer_type_{composite_layer};
      
      //
      // Structure of the composite neural network
      std::list< std::shared_ptr< NeuralNetwork > > nn_composite_;

      //
      // Cuda error status
      //cudaError_t cuda_err_{ cudaSuccess };

      
      //
      // Weights
      // weights on the host
      Weights  weights_;
      // number of weights
      int      number_of_weights_{0};
      // The weight indexes is an array of indexes where starts the weights of this layer
      std::vector< int >  weight_indexes_;
    };
}
#endif
