#ifndef ALPSFULLYCONNECTEDLAYER_H
#define ALPSFULLYCONNECTEDLAYER_H
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
  /** \class FullyConnectedLayer
   *
   * \brief 
   * FullyConnectedLayer class represents the basic layer element that can be used 
   * into a densly connected neural network.
   * 
   */
  template< typename ActivationFunction,
	    typename Weights,
	    typename CostFunction,
	    int Dim  >
  class FullyConnectedLayer : public Alps::Layer,
			      public Alps::Mountain
  {
    //
    //
    using Self = FullyConnectedLayer< ActivationFunction, Weights, CostFunction, Dim >;
    //
    // 
  public:
    /** Constructor. */
    explicit FullyConnectedLayer( const std::string,
				  const std::vector< std::size_t > );
    /** Destructor */
    virtual ~FullyConnectedLayer(){};

    
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
      {return fc_layer_size_;};
    // attach the next layer
    virtual       void                     set_next_layer( std::shared_ptr< Alps::Layer > Next ) override
    { next_layer_.push_back( Next );};


    //
    // Functions
    //
    // Add previous layer
    virtual       void                     add_layer( std::shared_ptr< Alps::Layer > Layer )     override
    { prev_layer_.push_back( Layer );};
    // Forward propagation
    virtual       void                     forward( std::shared_ptr< Alps::Climber > )           override;
    // Backward propagation
    virtual       void                     backward( std::shared_ptr< Alps::Climber > )          override;
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
    std::string                                         layer_name_{"__Fully_connected_layer__"};
      
    //
    // number of fully connected layers
    std::vector< std::size_t >                          fc_layer_size_;
    // enumeration for the naming convention
    enum Act
      {
       UNKNOWN        = -1,
       ACTIVATION     =  0,
       DERIVATIVE     =  1,
       ERROR          =  2,
       WEIGHTED_ERROR =  3
      };
    // Activation tuple (<0> - activation; <1> - derivative; <2> - error; <3> - weighted error)
    std::tuple< std::shared_ptr< double >,
		std::shared_ptr< double >,
		std::shared_ptr< double >,
		std::shared_ptr< double > >             current_activation_;

    //
    // Previous  layers information
    std::vector< std::shared_ptr< Alps::Layer > >       prev_layer_;
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
  template< typename AF, typename W, typename C, int D   >
  FullyConnectedLayer< AF, W, C, D >::FullyConnectedLayer( const std::string              Layer_name,
							   const std::vector<std::size_t> Fc_layer_size ):
    layer_name_{Layer_name}, fc_layer_size_{Fc_layer_size}
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
	if ( Fc_layer_size.size() != 1 )
	  throw MAC::MACException( __FILE__, __LINE__,
				   "Alps does not handle multiple layer in one Layer yet, except for the input layer.",
				   ITK_LOCATION );
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
  template< typename AF, typename W, typename C, int D   > void
  FullyConnectedLayer< AF, W, C, D >::forward( std::shared_ptr< Alps::Climber > Sub )
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
	// We get the number of previous layers attached to this layer. In this first loop,
	// we collect the number of nodes if the weights were not initialized
	// if the prev layer is nullptr, it represents the input data.
	std::cout << "Layer: " << layer_name_ << std::endl;
	if ( weights_.size() == 0 )
	  {
	    // If the weights were not initialized yet
	    for ( auto layer : prev_layer_ )
	      {
		std::string name = "__input_layer__";
		if ( layer )
		  {
		    name = layer->get_layer_name();
		    //
		    std::cout
		      << "Connected to: " << name
		      << " with " << layer->get_layer_size()[0] << " nodes" << std::endl;
		    //
		    weights_[name] = std::make_shared< W >( std::shared_ptr< FullyConnectedLayer< AF, W, C, D > >( this ),
							    fc_layer_size_,
							    layer->get_layer_size() );
		  }
		else
		  {
		    std::cout
		      << "Connected to: " << name
		      << " with " << subject->get_layer_size( name )[0] << " nodes" << std::endl;
		    //
		    weights_[name] = std::make_shared< W >( std::shared_ptr< FullyConnectedLayer< AF, W, C, D > >( this ),
							    fc_layer_size_,
							    subject->get_layer_size( name ) );
		  }
	      }
	  }
	
	/////////////////
	// Activations //
	/////////////////
	//
	//
	std::size_t layer_size = fc_layer_size_[0];
	// activation function
	std::shared_ptr< double > z     = std::shared_ptr< double >( new  double[layer_size],
								     std::default_delete< double[] >() );
	// Derivative of the activation function
	std::shared_ptr< double > dz    = std::shared_ptr< double >( new  double[layer_size],
								     std::default_delete< double[] >() );
	// Error back propagated in building the gradient
	std::shared_ptr< double > error = std::shared_ptr< double >( new  double[layer_size],
								     std::default_delete< double[] >() );
	// Weighted error back propagated in building the gradient
	std::shared_ptr< double > werr  = std::shared_ptr< double >( new  double[layer_size],
								     std::default_delete< double[] >() );
	// initialize to 0
	for ( std::size_t s = 0 ; s < layer_size ; s++ )
	  {
	    z.get()[s]     = 0.;
	    dz.get()[s]    = 0.;
	    error.get()[s] = 0.;
	    werr.get()[s]  = 0.;
	  }
	// We concaten the tensors from any layer connected to this layer
	for ( auto layer : prev_layer_ )
	  {
	    std::string name = "__input_layer__";
	    if ( layer )
	      // If the pointer exist, this is not the input layer
	      name = layer->get_layer_name();
	    //
	    // activation tuple (<0> - activation; <1> - derivative)
	    auto tuple = weights_[name]->activate( subject->get_layer(name) );
	    //
	    for ( std::size_t s = 0 ; s < layer_size ; s++ )
	      {
		z.get()[s]  += std::get< Act::ACTIVATION >(tuple).get()[s];
		dz.get()[s] += std::get< Act::DERIVATIVE >(tuple).get()[s];
	      }
	  }
	//
	// Get the activation tuple (<0> - activation; <1> - derivative; <2> - error)
	current_activation_ = std::make_tuple( z, dz, error, werr );

	
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
	    if( size_target[0] != fc_layer_size_[0])
	      {
		std::string
		  mess = "The target (" + std::to_string( size_target[0] );
		mess  += ") and the fit (" + std::to_string( fc_layer_size_[0] );
		mess  += ") does not match.";
		throw MAC::MACException( __FILE__, __LINE__,
					 mess.c_str(),
					 ITK_LOCATION );
	      }
	    //
	    // Cost function. 
	    C cost;
	    // It returns the error at the image level
	    std::get< Act::ERROR >( current_activation_ ) = cost.dL( (std::get< Act::ACTIVATION >( current_activation_ )).get(),
								     target.get_tensor().get(),
								     (std::get< Act::DERIVATIVE >( current_activation_ )).get(),
								     fc_layer_size_[0] );
	    // Save the energy for this image
	    double energy = cost.L( (std::get< Act::ACTIVATION >( current_activation_ )).get(),
				    target.get_tensor().get(),
				    fc_layer_size_[0] );
	    // record the energy for the image
	    subject->set_energy( energy );
	  }
	//
	// If the layer does not exist, for the image, it creates it.
	// Otherwise, it replace teh values from the last epoque and save the previouse epoque.
	subject->add_layer( layer_name_, fc_layer_size_,
			    current_activation_ );
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
  template< typename AF, typename W, typename C, int D   > void
  FullyConnectedLayer< AF, W, C, D >::backward( std::shared_ptr< Alps::Climber > Sub )
  {
    try
      {
	//
	// Down to subject
	std::shared_ptr< Alps::Subject< D > > subject = std::dynamic_pointer_cast< Alps::Subject< D > >(Sub);
	std::cout << "Layer backwards: " << layer_name_ << std::endl;

	//
	// If we don't have any next layer, we are at the last layer
	if ( next_layer_.size() == 0 )
	  {
	    std::cout
	      << "We are in the last layer: " << layer_name_
	      << " with " << fc_layer_size_[0] << " nodes" << std::endl;
	    //
	    // 
//	    for ( auto layer_weights = weights_.begin(); it != weights_.end(); ++it )
//	      {
//		//
//		// We have to create a way to get the vector and cut it to the right dimension
//		// of all the modalities that are going to be transfered to the layers attached
//		// to this layer.
//		subject->get_layer( (*layer_weights).first );  // which is a vector< Alps::LayerTensors< T, 2 > >
//		  (*layer_weights).second->weighted_error( std::get< Act::ERROR >(current_activation_) );
//		
//
	  }

	    
	  }
	else
	  for ( auto layer : next_layer_ )
	    {
	      std::string name = "__output_layer__";
	      if ( layer )
		{
		  name = layer->get_layer_name();
		  //
		  std::cout
		    << "Back connected from: " << name
		    << " with " << layer->get_layer_size()[0] << " nodes" << std::endl;
		}
	      else
		throw MAC::MACException( __FILE__, __LINE__,
				   "We should be connected to another layer.",
				   ITK_LOCATION );
	    }
	
	////////////////////////
	// Create the weights //
	////////////////////////
	//
	// We get the number of previous layers attached to this layer. In this first loop,
	// we collect the number of nodes if the weights were not initialized
//	//
//	// Prepare the backward variables
//	// Previous layer as alreay been saveed from the forward function
//	// current layer
//	weights_->set_current_dactivation();
//	// next layer
//	weights_->set_next_weights();
//
//	//
//	// Process the image error for the current layer
//
//	//
//	// Update the weights
//	weights_->update();
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit(-1);
      }
  };
}
#endif
