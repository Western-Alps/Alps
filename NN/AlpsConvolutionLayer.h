/*=========================================================================
* Alps is a deep learning library approach customized for neuroimaging data 
* Copyright (C) 2021 Yann Cobigo (yann.cobigo@yahoo.com)
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <https://www.gnu.org/licenses/>.
*=========================================================================*/
#ifndef ALPSCONVOLUTIONLAYER_H
#define ALPSCONVOLUTIONLAYER_H
//
//
//
#include <iostream>
#include <random>
#include <memory>
//
#include "MACException.h"
#include "AlpsLayer.h"
#include "AlpsMountain.h"
#include "AlpsBaseFunction.h"
#include "AlpsSubject.h"
#include "AlpsLayerTensors.h"

//
//
//
namespace Alps
{
  /** \class ConvolutionLayer
   *
   * \brief 
   * ConvolutionLayer class represents the basic layer element that can be used 
   * into a convolutional neural network.
   * 
   */
  template< typename ActivationFunction,
	    typename Weights,
	    typename Kernel,
	    typename CostFunction,
	    int Dim  >
  class ConvolutionLayer : public Alps::Layer,
			   public Alps::Mountain
  {
    //
    //
    using Self = ConvolutionLayer< ActivationFunction, Weights, Kernel, CostFunction, Dim >;
    //
    // 
  public:
    /** Constructor. */
    explicit ConvolutionLayer( const std::string, std::shared_ptr< Kernel > );
    /** Destructor */
    virtual ~ConvolutionLayer() = default;

    
    //
    // Accessors
    //
    // get the layer identification
    virtual const std::size_t              get_layer_id() const                                  override
    { return layer_id_;}	           
    // get the layer name	           
    virtual const std::string              get_layer_name() const                                override
    { return layer_name_;}	           
    // get number of weights	           
    virtual const int                      get_number_weights() const                            override
    { return 0.;};
    // get the layer size
    virtual const std::vector<std::size_t> get_layer_size() const                                override
    {return std::vector<std::size_t>();};
    // attach the next layer
    virtual       void                     set_next_layer( std::shared_ptr< Alps::Layer > Next ) override
    { next_layer_.push_back( Next );};


    //
    // Functions
    //
    // Add previous layer
    virtual       void                     add_layer( std::shared_ptr< Alps::Layer > )           override;
    // Forward propagation
    virtual       void                     forward( std::shared_ptr< Alps::Climber > )           override;
    // Backward propagation
    virtual       void                     backward( std::shared_ptr< Alps::Climber > )          override;
    // Update the weights at the end of the epoque
    virtual       void                     weight_update( std::shared_ptr< Alps::Climber > )     override;
    // Save the weights at the end of the epoque
    virtual       void                     save_weights( std::ofstream& ) const                  override;
    //
    //
    // Attach observers that need to be updated
    virtual       void                     attach( std::shared_ptr< Alps::Climber > )            override {};
    // Notify the observers for updates
    virtual       void                     notify()                                              override {};
    // Save the weights at the end of the epoque
    virtual       void                     save_weight_file( const std::size_t ) const           override{};

    
  private:
    // layer unique ID
    std::size_t                                         layer_id_{0};
    // Layer's name
    std::string                                         layer_name_{"__Convolution_layer__"};
      
    //
    // Convolution window
    std::shared_ptr< Kernel >                           convolution_window_{nullptr};
    // enumeration for the naming convention
    enum Act
      {
       UNKNOWN        = -1,
       ACTIVATION     =  0,
       DERIVATIVE     =  1,
       ERROR          =  2,
       WEIGHTED_ERROR =  3
      };

    //
    // Previous  layers information
    std::map< /* Layer_name */ std::string,
	      std::shared_ptr< Alps::Layer > >          prev_layer_;
    // Next layers information
    std::vector< std::shared_ptr< Alps::Layer > >       next_layer_;
    //
    // Observers
    // Observers containers
    std::vector< Weights >                              weights_;
  };
  //
  //
  //
  template< typename AF, typename W, typename K, typename C, int D >
  ConvolutionLayer< AF, W, K, C, D >::ConvolutionLayer( const std::string    Layer_name,
							std::shared_ptr< K > Convolution_window ):
    layer_name_{Layer_name}, convolution_window_{Convolution_window}
  {
    try
      {
	//
	// Create a unique id for the layer
	std::random_device                   rd;
	std::mt19937                         generator( rd() );
	std::uniform_int_distribution< int > distribution( 0, 1UL << 16 );
	//
	layer_id_ = distribution( generator );
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit(-1);
      }
  };
  //
  //
  //
  template< typename AF, typename W, typename K, typename C, int D > void
  ConvolutionLayer< AF, W, K, C, D >::add_layer( std::shared_ptr< Alps::Layer > Layer )
  {
    try
      {
	if ( Layer )
	  prev_layer_[Layer->get_layer_name()] = Layer;
	else
	  prev_layer_["__input_layer__"] = nullptr;
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit(-1);
      }
  };
  //
  //
  //
  template< typename AF, typename W, typename K, typename C, int D > void
  ConvolutionLayer< AF, W, K, C, D >::forward( std::shared_ptr< Alps::Climber > Sub )
  {
    try
      {
	//
	// Down to subject
	std::shared_ptr< Alps::Subject< D > > subject = std::dynamic_pointer_cast< Alps::Subject< D > >(Sub);
	
	////////////////////////
	// Create the weights //
	////////////////////////
	//
	// We get features or inputs from previous layers attached to this layer. 
	// if the prev layer is nullptr, it represents the input data.
	//std::cout << "Layer: " << layer_name_ << std::endl;
	std::vector< std::shared_ptr< Alps::LayerTensors< double, D > > > attached_layers;
	for ( auto layer : prev_layer_ )
	  {
	    std::string name = "__input_layer__";
	    if ( layer.second )
	      {
		name = layer.first;
		//std::cout << "Connected to: " << name << std::endl;
	      }
	    //else
	    //  std::cout << "Connected to: " << name << std::endl;
	    //
	    attached_layers.insert( attached_layers.end(),
				    subject->get_layer(name).begin(), subject->get_layer(name).end() );
	  }
	//
	// Create the weights for the k features
	int kernels = convolution_window_->get_number_kernel();
	mtx_.lock();
	if ( weights_.size() != static_cast< std::size_t >(kernels) )
	  for ( int k = 0 ; k < kernels ; k++ )
	    weights_.push_back( W(*this, convolution_window_, k) );
	// Initialize the weights matrix if not yet done
	if ( !convolution_window_->initialized() )
	  convolution_window_->get_image_information( ( *(attached_layers[0].get()) ).get_image(TensorOrder1::ACTIVATION).get_image_region() );
	mtx_.unlock();
	// Every layer attached to this layer should have exactly the same dimensions
	std::size_t tot_features    = attached_layers.size();
	std::size_t prev_layer_size = ( *(attached_layers[0].get()) ).get_image(TensorOrder1::ACTIVATION).get_tensor_size()[0];
	//
	for ( std::size_t feature = 1 ; feature < tot_features ; feature++ )
	  if ( prev_layer_size != ( *(attached_layers[feature].get()) ).get_image(TensorOrder1::ACTIVATION).get_tensor_size()[0] )
	    throw MAC::MACException( __FILE__, __LINE__,
				     "All attached layers must have the same output features' dimensions.",
				     ITK_LOCATION );
	
	
	/////////////////
	// Activations //
	/////////////////
	//
	// Loop over the current layer K kernels
	// All the features from the previous layers, generated by one image, is going to be
	// processed by the same kernel K.
	for ( int k = 0 ; k < kernels ; k++ )
	  {
	    // Get the activation tuple:
	    // <0> - activation;
	    // <1> - derivative;
	    // <2> - error;
	    // <3> - weighted error
	    auto tuple   = weights_[k].activate( attached_layers );
	    // get the layer size to initialize the other tensors
	    std::size_t layer_size = tuple[ Act::ACTIVATION ].size();
	    std::array< std::vector< double >, 4 > current_activation = { std::move( tuple[ Act::ACTIVATION ] ),
									  std::move( tuple[ Act::DERIVATIVE ] ),
									  std::vector< double >( layer_size, 0. ) /* error */,
									  std::vector< double >( layer_size, 0. ) /* werr */ };
	    
	    //////////////////////////////////////
	    // Save the activation information //
	    /////////////////////////////////////
	    //
	    // If the layer does not exist, for the image, it creates the layer.
	    // Otherwise, it replace the values from the last epoque and save the previouse epoque.
	    subject->add_layer( layer_name_, k,
				convolution_window_->get_output_image_dimensions(),
				current_activation );
	  }
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit(-1);
      }
  };
  //
  //
  //
  template< typename AF, typename W, typename K, typename C, int D > void
  ConvolutionLayer< AF, W, K, C, D >::backward( std::shared_ptr< Alps::Climber > Sub )
  {
    try
      {
	//
	// Down to subject
	std::shared_ptr< Alps::Subject< D > > subject = std::dynamic_pointer_cast< Alps::Subject< D > >(Sub);
	//std::cout << "Layer backwards: " << layer_name_ << std::endl;
	// get the activation tuple
	auto image_tensors = subject->get_layer( layer_name_ );
	

	/////////////////////
	// Weighted error //
	////////////////////
	//
	// Process the weighted error for the previous layers
	// The latest layer weighted error should already be processed
	int kernels = convolution_window_->get_number_kernel();
	for ( int k = 0 ; k < kernels ; k++ )
	  for ( auto layer_weights : prev_layer_ )
	    {
	      //
	      std::string name = layer_weights.first;
	      //std::cout << "weights of layer: " << name << std::endl;
	      weights_[k].weighted_error( subject->get_layer( name ),
					  image_tensors );
	    }


	////////////////////////
	// Update the weights //
	////////////////////////
	//
	for ( int k = 0 ; k < kernels ; k++ )
	  for ( auto layer_weights : prev_layer_ )
	    {
	      std::string name = layer_weights.first;
	      weights_[k].set_activations( subject->get_layer( name ),
					   image_tensors );
	      weights_[k].update();
	    }
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit(-1);
      }
  };
  //
  //
  //
  template< typename AF, typename W, typename K, typename C, int D > void
  ConvolutionLayer< AF, W, K, C, D >::weight_update( std::shared_ptr< Alps::Climber > Sub )
  {
    try
      {
	//
	// Down to subject
	std::shared_ptr< Alps::Subject< D > > subject = std::dynamic_pointer_cast< Alps::Subject< D > >(Sub);
	//std::cout << "forcing the weight update: " << layer_name_ << std::endl;

	////////////////////////
	// Update the weights //
	////////////////////////
	//
	int kernels = convolution_window_->get_number_kernel();
	//
	for ( int k = 0 ; k < kernels ; k++ )
	  for ( auto layer_weights : prev_layer_ )
	    weights_[k].update();
//	for ( auto layer_weights : weights_ )
//	  {
//	    std::string name = layer_weights.first;
//	    weights_[name]->update();
//	  }
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit(-1);
      }
  };
  //
  //
  //
  template< typename AF, typename W, typename K, typename C, int D > void
  ConvolutionLayer< AF, W, K, C, D >::save_weights( std::ofstream& Weights_file  ) const
  {
    try
      {
	//
	// Name of the layer
	Weights_file.write( layer_name_.c_str(), sizeof(char)*layer_name_.size() );
	// Then the weights
	convolution_window_->save_weights( Weights_file  );
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit(-1);
      }
  };
}
#endif
