#ifndef ALPSTRANSPOSEDCONVOLUTIONLAYER_H
#define ALPSTRANSPOSEDCONVOLUTIONLAYER_H
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
#include "AlpsConvolutionLayer.h"

//
//
//
namespace Alps
{
  /** \class TransposedConvolutionLayer
   *
   * \brief 
   * TransposedConvolutionLayer class represents the basic layer element that can be used 
   * into a convolutional neural network as a deconvolution process.
   * 
   */
  template< typename ActivationFunction,
	    typename Weights,
	    typename Kernel,
	    typename CostFunction,
	    int Dim  >
  class TransposedConvolutionLayer : public Alps::Layer,
				     public Alps::Mountain
  {
    //
    //
    using Self = TransposedConvolutionLayer< ActivationFunction, Weights, Kernel, CostFunction, Dim >;
    //
    // 
  public:
    /** Constructor. */
    TransposedConvolutionLayer( const std::string, std::shared_ptr< Kernel > );
    /** Destructor */
    virtual ~TransposedConvolutionLayer(){};

    
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
    std::string                                         layer_name_{"__Transposed_convolution_layer__"};
      
    //
    // TransposedConvolution window
    std::shared_ptr< Kernel >                           convolution_window_;
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
    std::map< /* Layer_name */ std::string,
	       std::shared_ptr< Weights > >              weights_;
  };
  //
  //
  //
  template< typename AF, typename W, typename K, typename C, int D >
  TransposedConvolutionLayer< AF, W, K, C, D >::TransposedConvolutionLayer( const std::string    Layer_name,
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
  TransposedConvolutionLayer< AF, W, K, C, D >::add_layer( std::shared_ptr< Alps::Layer > Layer )
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
  TransposedConvolutionLayer< AF, W, K, C, D >::forward( std::shared_ptr< Alps::Climber > Sub )
  {
    try
      {
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
  TransposedConvolutionLayer< AF, W, K, C, D >::backward( std::shared_ptr< Alps::Climber > Sub )
  {
    try
      {
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
  TransposedConvolutionLayer< AF, W, K, C, D >::weight_update( std::shared_ptr< Alps::Climber > Sub )
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
