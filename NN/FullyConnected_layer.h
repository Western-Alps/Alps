#ifndef FULLYCONNECTED_LAYER_H
#define FULLYCONNECTED_LAYER_H
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
//
//
#include "MACException.h"
#include "MACLoadDataSet.h"
#include "Subject.h"
#include "NeuralNetwork.h"
#include "Weights.h"
//
// CUDA
//
#include <cuda_runtime.h>
#include "FullyConnected_layer_CUDA.cuh"
///
//
//
namespace MAC
{

  /** \class FullyConnected_layer
   *
   * \brief 
   * 
   * 
   */
  class FullyConnected_layer : public NeuralNetwork
    {
      //
      // Some typedef
      using Image3DType = itk::Image< double, 3 >;
      using Reader3D    = itk::ImageFileReader< Image3DType >;

    public:
      /** Constructor. */
      FullyConnected_layer( const std::string, const int,
			    const int,         const int* );
      //
      //explicit Subject( const int, const int );

      /** Destructor */
      virtual ~FullyConnected_layer();

      //
      // Initialization
      virtual void initialization();
      //
      // get the layer name
      virtual std::string get_layer_name(){ return layer_name_;};
      //
      // get the layer name
      virtual Layer get_layer_type(){ return fully_connected_layer;};
      //
      // Forward propagation
      virtual void forward( Subject&, const Weights& W = Weights() );
      //
      // Backward propagation
      virtual void backward();
      //
      // Backward error propagation to CNN
      virtual void backward_error_propagation();
      //
      //
      virtual void add( std::shared_ptr< NeuralNetwork > ){};
      //
      //
      virtual int get_number_weights() const { return number_of_weights_;};

    private:
      //
      // Private constructor
      void init_();
      
      //
      // Convolutional layer's name
      std::string layer_name_;
      
      //
      // Weights
      const int  layer_number_;
      // number of fully connected layers
      const int  number_fc_layers_;
      // 
      int* fc_layers_;
      bool initializarion_done_{false};
      int  number_of_weights_{0};
      //
      double* weights_{nullptr};

      //
      // Neurons, activations and delta
      using  Neurons_type = std::tuple< std::vector< std::shared_ptr<double> > /* activations */,
	                                std::vector< std::shared_ptr<double> > /* neurons */,
	                                std::vector< std::shared_ptr<double> > /* deltas */  >;
      //
      std::map< std::string, Neurons_type > neurons_;
      
      //
      // Cuda treatment
      FullyConnected_layer_CUDA cuda_bwd_{ FullyConnected_layer_CUDA() };
    };
}
#endif
