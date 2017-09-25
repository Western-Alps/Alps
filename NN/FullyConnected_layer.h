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
// CUDA
//
#include <cuda_runtime.h>
//
//
//
#include "MACException.h"
#include "Subject.h"
#include "NeuralNetwork.h"
//
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
      //
      virtual void backward(){};
      //
      //
      virtual void add( std::shared_ptr< NeuralNetwork > ){};
      //
      //
      virtual int get_number_weights() const { return 1;};

    private:
      //
      // Convolutional layer's name
      std::string layer_name_;
      
      //
      // Weights
      const int  layer_number_;
      // number of fully connected layers
      const int  number_fc_layers_;
      // 
      const int* fc_layers_;
      int        number_of_weights_{0};

      //
      // Neurons
      int     number_of_neurons_{0};
      double* activation_;
      double* neurons_;
      
      //
      //
      // Measures grouped in vector of 3D image
      // Convolution image: vector for each modality
      std::vector< Image3DType::Pointer > convolution_images_;
      // Pulling image: vector for each modality
      std::vector< Image3DType::Pointer > pull_images_;
    };
}
#endif
