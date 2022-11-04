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
#ifndef ALPSRECONSTRUCTIONLAYER_H
#define ALPSRECONSTRUCTIONLAYER_H
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
  /** \class ReconstructionLayer
   *
   * \brief 
   * ReconstructionLayer class combined information from multiple convolutional
   * feature maps layers to build the cost function. 
   * 
   */
  template< typename ActivationFunction,
	    typename Weights,
	    typename CostFunction,
	    int Dim  >
  class ReconstructionLayer : public Alps::Layer,
			      public Alps::Mountain
  {
    //
    //
    using Self = ReconstructionLayer< ActivationFunction, Weights, CostFunction, Dim >;
    //
    // 
  public:
    /** Constructor. */
    explicit ReconstructionLayer( const std::string );
    /** Destructor */
    virtual ~ReconstructionLayer() = default;

    
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
    //
    //
    // Attach observers that need to be updated
    virtual       void                     attach( std::shared_ptr< Alps::Climber > )            override {};
    // Notify the observers for updates
    virtual       void                     notify()                                              override {};

    
  private:
    // layer unique ID
    std::size_t                                         layer_id_{0};
    // Layer's name
    std::string                                         layer_name_{"__Reconstruction_layer__"};
      
    //
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
    std::shared_ptr< Weights >                          weights_{nullptr};
    //
    // Cumulated error. Error cumulated for each image
    std::vector< double >                               cumulated_error_;
    std::size_t                                         number_images_{0};
  };
  //
  //
  //
  template< typename AF, typename W, typename C, int D >
  ReconstructionLayer< AF, W, C, D >::ReconstructionLayer( const std::string Layer_name ):
    layer_name_{Layer_name}
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
  template< typename AF, typename W, typename C, int D > void
  ReconstructionLayer< AF, W, C, D >::add_layer( std::shared_ptr< Alps::Layer > Layer )
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
  template< typename AF, typename W, typename C, int D > void
  ReconstructionLayer< AF, W, C, D >::forward( std::shared_ptr< Alps::Climber > Sub )
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
	std::vector< std::reference_wrapper< Alps::LayerTensors< double, D > > > attached_layers;
	for ( auto layer : prev_layer_ )
	  {
	    std::string name = "__input_layer__";
	    if ( layer.second )
	      {
		name = layer.first;
		//std::cout << "Connected to: " << name << std::endl;
	      }
	    else
	      std::cout << "Connected to: " << name << std::endl;
	    //
	    attached_layers.insert( attached_layers.end(),
				    subject->get_layer(name).begin(), subject->get_layer(name).end() );
	  }
	//
	// Make sure the features have the same image dimensions.
	std::size_t tot_features = attached_layers.size();
	std::size_t layer_size   = (attached_layers[0].get()).get_image(TensorOrder1::ACTIVATION).get_tensor_size()[0];
	//
	for ( std::size_t feature = 1 ; feature < tot_features ; feature++ )
	  if ( layer_size != (attached_layers[feature].get()).get_image(TensorOrder1::ACTIVATION).get_tensor_size()[0] )
	    throw MAC::MACException( __FILE__, __LINE__,
				     "All attached layers must have the same dimensions for the output features.",
				     ITK_LOCATION );

	
	/////////////////
	// Activations //
	/////////////////
	//
	// Check the weights were created
	if ( !weights_ )
	  weights_ = std::make_shared< W >( *this );
	//
	// Get the activation tuple (<0> - activation; <1> - derivative; <2> - error; <3> - weighted error))
	auto tuple = weights_->activate( attached_layers );
	std::array< std::vector< double >, 4 > current_activation = { tuple[ Act::ACTIVATION ],
								      tuple[ Act::DERIVATIVE ],
								      std::vector< double >( layer_size, 0. ) /* error */,
								      std::vector< double >( layer_size, 0. ) /* werr */ };
	    
	
	//////////////////////////////////////
	// Save the activation information //
	/////////////////////////////////////
	//
	// If we are at the last level, we can estimate the error of the image target
	// with the fit
	if ( layer_name_ == "__output_layer__" )
	  {
	    //
	    // Get image target from subject
	    auto target = subject->get_target();
	    // Get the size of the target and compare to the fit
	    std::vector< std::size_t > size_target = target.get_tensor_size();
	    // Check we are comparing the same thing
	    if( size_target[0] != layer_size )
	      {
		std::string
		  mess = "The target (" + std::to_string( size_target[0] );
		mess  += ") and the fit (" + std::to_string( layer_size );
		mess  += ") does not match.";
		throw MAC::MACException( __FILE__, __LINE__,
					 mess.c_str(),
					 ITK_LOCATION );
	      }
	    //
	    // Cost function. 
	    C cost;
	    // Returns the error at the image level
	    current_activation[Act::ERROR] = std::move( cost.dL(current_activation[Act::ACTIVATION],
								target.get_tensor(),
								current_activation[Act::DERIVATIVE],
								layer_size) );
	    //
	    // Cumulate the error through all images
	    if ( cumulated_error_.empty() )
	      cumulated_error_.resize( layer_size, 0. );
	    // Account for this specific subject
	    std::transform( cumulated_error_.begin(), cumulated_error_.end(),
			    current_activation[Act::ERROR].begin(), cumulated_error_.begin(),
			    std::plus< double >() );
	    number_images_++;
//	    for ( std::size_t i = 0 ; i < layer_size ; i++ )
//	      std::cout
//		<< "cumulated_error_["<<i<<"] = " << cumulated_error_[i] 
//		<< " for image " << number_images_
//		<< std::endl;
	    //
	    // Save the energy for this image
	    double energy = cost.L( current_activation[Act::ACTIVATION],
				    target.get_tensor(),
				    layer_size );
	    // record the energy for the image
	    subject->set_energy( energy );
	  }
	else
	  {
	    std::string
	      mess = "Reconstruction Layer has not been thought for another type of application yet. ";
	    mess  += "It has to be used as a __output_layer__";
	    throw MAC::MACException( __FILE__, __LINE__,
				     mess.c_str(),
				     ITK_LOCATION );
	  }
	
	//
	// If the layer does not exist, for the image, it creates it.
	// Otherwise, it replace the values from the last epoque and save the previouse epoque.
	subject->add_layer( layer_name_, 
			    {layer_size},
			    current_activation );
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
  template< typename AF, typename W, typename C, int D > void
  ReconstructionLayer< AF, W, C, D >::backward( std::shared_ptr< Alps::Climber > Sub )
  {
    try
      {
	//
	// Down to subject
	std::shared_ptr< Alps::Subject< D > > subject = std::dynamic_pointer_cast< Alps::Subject< D > >(Sub);
	//std::cout << "Layer backwards: " << layer_name_ << std::endl;
	// get the activation tuple
	auto image_tensors = subject->get_layer( layer_name_ );

	//
	// Average the cumulated error and replace with the one from the subject
	for( double& cumul : cumulated_error_ ) 
	  cumul /= static_cast< double >( number_images_ );
	// Attached the averaged errors to the last subject
	(image_tensors[0].get())[TensorOrder1::ERROR] = std::move( cumulated_error_ );
	// Then reset the layer elements
	cumulated_error_.clear();
	number_images_ = 0;


	//
	// If we don't have any next layer, we are at the last layer
	//std::cout << "We are in the layer: " << layer_name_ << std::endl;


	/////////////////////
	// Weighted error //
	////////////////////
	//
	// Process the weighted error for the previous layer
	// The latest layer weighted error should already be processed
	for ( auto layer_weights : prev_layer_ )
	  {
	    std::string name = layer_weights.first;
	    //std::cout << "weights of layer: " << name << std::endl;
	    weights_->weighted_error( subject->get_layer( name ),
				      image_tensors );
	  }


	////////////////////////
	// Update the weights //
	////////////////////////
	//
	weights_->set_activations( image_tensors,
				   image_tensors /*Dummy setting*/ );
	weights_->update();
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
  template< typename AF, typename W, typename C, int D > void
  ReconstructionLayer< AF, W, C, D >::weight_update( std::shared_ptr< Alps::Climber > Sub )
  {
    try
      {
//	//
//	// Down to subject
//	std::shared_ptr< Alps::Subject< D > > subject = std::dynamic_pointer_cast< Alps::Subject< D > >(Sub);
//	std::cout
//	  << "forcing the weight update: " << layer_name_ << std::endl;
//
//	////////////////////////
//	// Update the weights //
//	////////////////////////
//	//
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
}
#endif
