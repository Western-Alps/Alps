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
#include "AlpsTools.h"
// Gradients
#include "AlpsGradient.h"
#include "AlpsSGD.h"
#include "AlpsAdaGrad.h"
// Weights
#include "Weights.h"
#include "Convolutional_window.h"
#include "Deconvolutional_window.h"
//
// Layers
#include "Convolution.h"
#include "Convolutional_layer.h"
#include "ConvolutionalAutoEncoder_layer.h"
#include "FullyConnected_layer.h"
#include "NN_test.h"
//
#include "AlpsFullyConnectedLayer.h"
// Activation
#include "Activations.h"
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
      // get the layer name
      virtual double get_energy(){ return energy_;};
      //
      // Forward propagation
      virtual void forward( Subject& Sub, const Weights& W = Weights() )
      {
	//
	for ( auto nn_elem : nn_composite_ )
	  {
	    std::cout << "New elem" << std::endl;
	    nn_elem->forward( Sub, weights_ );
	    // we just want to save the last energy
	    // energies are already cumulated in the container
	    std::cout << "E = " << nn_elem->get_energy() << std::endl;
	    energy_ = nn_elem->get_energy();
	  }
      };
      //
      //
      virtual void backward()
      {
	// 1. Reset energy cost function
	// 2. propagate
	std::list< std::shared_ptr< NeuralNetwork > >::reverse_iterator rit = nn_composite_.rbegin();
	for ( ; rit != nn_composite_.rend() ; rit++ )
	  {
	    std::cout << "Bkw elem" << std::endl;
	    (*rit)->backward();
	    (*rit)->backward_error_propagation();
	  }
      };
      //
      // Backward error propagation
      virtual void backward_error_propagation(){};
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
      double energy_{0.};
      
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
