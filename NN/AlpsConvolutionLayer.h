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
   * into a densly connected neural network.
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
    virtual ~ConvolutionLayer(){};

    
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
    std::string                                         layer_name_{"__Convolution_layer__"};
      
    //
    // Convolution window
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
    std::map< /* kernel number */ int,
	       std::shared_ptr< Weights > >              weights_;
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

	//
	//
	int kernels = convolution_window_->get_number_kernel();
	for ( int k = 0 ; k < kernels ; k++ )
	  weights_[k] = nullptr;
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
	// We get the number of previous layers attached to this layer. 
	// if the prev layer is nullptr, it represents the input data.
	// Gather the features from other layers.
	std::cout << "Layer: " << layer_name_ << std::endl;
	std::vector< Alps::LayerTensors< double, D > > attached_layers;
	for ( auto layer : prev_layer_ )
	  {
	    std::string name = "__input_layer__";
	    if ( layer.second )
	      {
		name = layer.first;
		//
		std::cout
		  << "Connected to: " << name << std::endl;
	      }
	    else
	      {
		std::cout
		  << "Connected to: " << name << std::endl;
	      }
	    //
	    attached_layers.insert( attached_layers.end(),
				    subject->get_layer(name).begin(), subject->get_layer(name).end() );
	  }
	//
	// Make sure the features have the same image dimensions.
	if ( convolution_window_->get_weights_matrix().nonZeros() == 0 )
	  {
	    // If the weights were not initialized yet
	    // ( const typename ImageType< D >::RegionType Region )
	    convolution_window_->get_image_information( attached_layers[0].get_image(TensorOrder1::ACTIVATION).get_image_region() );
	    // Every layer attached to this layer should have exactly the same dimensions
	    // from prev_layer_[0]
	    // ToDo --> Make sure to remember the dimension and the check all the layers have the same dims
	    
	  }
	
	/////////////////
	// Activations //
	/////////////////
	//
	// The layer_size, here, represents the size of the output image
	std::size_t layer_size = 1;
	for ( int d = 0 ; d < D ; d++ )
	  layer_size *= convolution_window_->get_output_image_dimensions()[d];
	//
	// Loop over the K kernels
	int kernels = convolution_window_->get_number_kernel();
	for ( int k = 0 ; k < kernels ; k++ )
	  {
	    //
	    // Check the weights were created
	    if ( !weights_[k] )
	      {
		weights_[k] = std::make_shared< W >( std::shared_ptr< ConvolutionLayer< AF, W, K, C, D > >( this ),
						     convolution_window_, k );
		
	      }
	    auto tuple = weights_[k]->activate( attached_layers );
//	    // activation function
//	    std::shared_ptr< double > z     = std::shared_ptr< double >( new  double[layer_size],
//									 std::default_delete< double[] >() );
//	    // Derivative of the activation function
//	    std::shared_ptr< double > dz    = std::shared_ptr< double >( new  double[layer_size],
//									 std::default_delete< double[] >() );
//	    // Error back propagated in building the gradient
//	    std::shared_ptr< double > error = std::shared_ptr< double >( new  double[layer_size],
//									 std::default_delete< double[] >() );
//	    // Weighted error back propagated in building the gradient
//	    std::shared_ptr< double > werr  = std::shared_ptr< double >( new  double[layer_size],
//									 std::default_delete< double[] >() );
//	    // initialize to 0
//	    for ( std::size_t s = 0 ; s < layer_size ; s++ )
//	      {
//		z.get()[s]     = 0.;
//		dz.get()[s]    = 0.;
//		error.get()[s] = 0.;
//		werr.get()[s]  = 0.;
//	      }
//	    // We concaten the tensors from any layer connected to this layer
//	    for ( auto layer : prev_layer_ )
//	      {
//		std::string name = "__input_layer__";
//		if ( layer.second )
//		  // If the pointer exist, this is not the input layer
//		  name = layer.first;
//		//
//		// activation tuple (<0> - activation; <1> - derivative)
//		auto tuple = weQights_[name]->activate( subject->get_layer(name) );
//		//
//		for ( std::size_t s = 0 ; s < layer_size ; s++ )
//		  {
//		    z.get()[s]  += std::get< Act::ACTIVATION >(tuple).get()[s];
//		    dz.get()[s] += std::get< Act::DERIVATIVE >(tuple).get()[s];
//		  }
//	      }
//	    //
//	    // Get the activation tuple (<0> - activation; <1> - derivative; <2> - error)
//	    std::tuple< std::shared_ptr< double >,
//			std::shared_ptr< double >,
//			std::shared_ptr< double >,
//			std::shared_ptr< double > > current_activation = std::make_tuple( z, dz, error, werr );
	  }
	    
	
//	//////////////////////////////////////
//	// Save the activation information //
//	/////////////////////////////////////
//	//
//	// If we are at the last level, we can estimate the error of the image target
//	// with the fit
//	if ( layer_name_ == "__output_layer__" )
//	  {
//	    //
//	    // Get image target from subject
//	    auto target = subject->get_target();
//	    // Get the size of the target and compare to the fit
//	    std::vector< std::size_t > size_target = target.get_tensor_size();
//	    // Check we are comparing the same thing
//	    if( size_target[0] != fc_layer_size_[0])
//	      {
//		std::string
//		  mess = "The target (" + std::to_string( size_target[0] );
//		mess  += ") and the fit (" + std::to_string( fc_layer_size_[0] );
//		mess  += ") does not match.";
//		throw MAC::MACException( __FILE__, __LINE__,
//					 mess.c_str(),
//					 ITK_LOCATION );
//	      }
//	    //
//	    // Cost function. 
//	    C cost;
//	    // Returns the error at the image level
//	    std::get< Act::ERROR >( current_activation ) = cost.dL( (std::get< Act::ACTIVATION >( current_activation )).get(),
//								     target.get_tensor().get(),
//								     (std::get< Act::DERIVATIVE >( current_activation )).get(),
//								     fc_layer_size_[0] );
//	    // Save the energy for this image
//	    double energy = cost.L( (std::get< Act::ACTIVATION >( current_activation )).get(),
//				    target.get_tensor().get(),
//				    fc_layer_size_[0] );
//	    // record the energy for the image
//	    subject->set_energy( energy );
//	  }
//	//
//	// If the layer does not exist, for the image, it creates it.
//	// Otherwise, it replace the values from the last epoque and save the previouse epoque.
//	subject->add_layer( layer_name_, fc_layer_size_,
//			    current_activation );
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
//	//
//	// Down to subject
//	std::shared_ptr< Alps::Subject< D > > subject = std::dynamic_pointer_cast< Alps::Subject< D > >(Sub);
//	std::cout << "Layer backwards: " << layer_name_ << std::endl;
//	// get the activation tuple
//	auto image_tensors = subject->get_layer( layer_name_ );
//	
//
//	//
//	// If we don't have any next layer, we are at the last layer
//	std::cout
//	  << "We are in the last layer: " << layer_name_
//	  << " with " << fc_layer_size_[0] << " nodes" << std::endl;
//
//
//	/////////////////////
//	// Weighted error //
//	////////////////////
//	//
//	// Process the weighted error for the previous layer
//	// The latest layer weighted error should already be processed
//	for ( auto layer_weights : weights_ )
//	  {
//	    std::cout << "weights of layer: " << layer_weights.first << std::endl;
//	    //
//	    std::string name = layer_weights.first;
//	    
//	    weights_[name]->weighted_error( subject->get_layer( name ),
//					    image_tensors );
//
//	  }
//
//
//	////////////////////////
//	// Update the weights //
//	////////////////////////
//	//
//	for ( auto layer_weights : weights_ )
//	  {
//	    std::string name = layer_weights.first;
//	    weights_[name]->set_activations( image_tensors,
//					     subject->get_layer( name ) );
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
  ConvolutionLayer< AF, W, K, C, D >::weight_update( std::shared_ptr< Alps::Climber > Sub )
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
