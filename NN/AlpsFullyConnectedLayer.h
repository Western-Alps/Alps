#ifndef ALPSFULLYCONNECTEDLAYER_H
#define ALPSFULLYCONNECTEDLAYER_H
//
//
//
#include <iostream>
#include <tuple>
#include <memory>
//
#include "MACException.h"
#include "AlpsLayer.h"
#include "AlpsMountain.h"
#include "AlpsFunction.h"
#include "AlpsSubject.h"
#include "AlpsImage.h"

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
  template< typename ActivationFunction, typename Weights, int Dim  >
  class FullyConnectedLayer : public Alps::Layer,
			      public Alps::Mountain
  {
    //
    //
    using Self = FullyConnectedLayer< ActivationFunction, Weights, Dim >;
    //
    // 
  public:
    /** Constructor. */
    explicit FullyConnectedLayer( const std::string,
				  const std::vector<int> );
    /** Destructor */
    virtual ~FullyConnectedLayer(){};

    
    //
    // Accessors
    //
    // get the layer name
    virtual const std::string      get_layer_name()                                      const override
    { return layer_name_;}
    // get number of weights
    virtual const int              get_number_weights()                                  const override
    { return 0.;};
    // get the layer size
    virtual const std::vector<int> get_layer_size()                                      const override
      {return fc_layer_size_;};
    // attach the next layer
    virtual       void             set_next_layer( std::shared_ptr< Alps::Layer > Next )       override
      { next_layer_ = Next;};


    //
    // Functions
    //
    // Add previous layer
    virtual       void             add_layer( std::shared_ptr< Alps::Layer > Layer )           override
    { prev_layer_.push_back( Layer );};
    // Forward propagation
    virtual       void             forward( std::shared_ptr< Alps::Climber > )                 override;
    // Backward propagation
    virtual       void             backward()                                                  override {};
    //
    //
    // Attach observers that need to be updated
    virtual       void             attach( std::shared_ptr< Alps::Climber > )                  override {};
    // Notify the observers for updates
    virtual       void             notify()                                                    override {};

    
  private:
    // Layer's name
    std::string                                   layer_name_{"__Fully_connected_layer__"};
    // Activation function
    std::shared_ptr< Alps::Function >             activation_func_{std::make_shared< ActivationFunction >()};
      
    //
    // number of fully connected layers
    std::vector< int >                            fc_layer_size_;

    //
    // Previous  layers information
    std::vector< std::shared_ptr< Alps::Layer > > prev_layer_;
    // Next layers information
    std::shared_ptr< Alps::Layer >                next_layer_;
    //
    // Observers
    // Observers containers
    std::shared_ptr< Weights >                    weights_{nullptr};
  };
  //
  //
  //
  template< typename AF, typename W, int D   >
  FullyConnectedLayer< AF, W, D >::FullyConnectedLayer( const std::string      Layer_name,
							const std::vector<int> Fc_layer_size ):
    layer_name_{Layer_name}, fc_layer_size_{Fc_layer_size}
  {
    try
      {
	if ( Fc_layer_size.size() != 1 )
	  throw MAC::MACException( __FILE__, __LINE__,
				   "Alps does not handle multiple layer in one Layer yet.",
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
  template< typename AF, typename W, int D   > void
  FullyConnectedLayer< AF, W, D >::forward( std::shared_ptr< Alps::Climber > Sub )
  {
    try
      {
	//
	// Down to subject
	std::shared_ptr< Alps::Subject< D > > subject = std::dynamic_pointer_cast< Alps::Subject< D > >(Sub);
	//
	// We get the number of previous layers attached to this layer. In this first loop,
	// we collect the number of nodes if the weights were not initialized
	std::cout << "Layer: " << layer_name_ << std::endl;
	std::vector< std::shared_ptr< Alps::Image< double, 2 > > > prev_layer_tensors;
	if ( !weights_ )
	  {
	    // If the weights were not initialized yet
	    std::vector< int > prev_layer_size;
	    for ( auto layer : prev_layer_ )
	      if ( layer )
		{
		  for ( auto mod : layer->get_layer_size() )
		    {
		      std::cout
			<< "Connected to: " << layer->get_layer_name()
			<< " with " << layer->get_layer_size()[0] << " nodes" << std::endl;
		      prev_layer_size.push_back( mod );
		    }
		}
	      else
		{
		  // the connected layer is the input layer
		  for ( auto mod : subject->get_layer_size() )
		    {
		      std::cout
			<< "Connected to: " << "__input_layer__"
			<< " with " << subject->get_layer_size()[0] << " nodes" << std::endl;
		      prev_layer_size.push_back( mod );
		      // Concate the input iamges
		      auto input_images = subject->get_layer_z("__input_layer__");
		      prev_layer_tensors.insert( prev_layer_tensors.end(),
						 input_images.begin(),
						 input_images.end() );
		    }
		}
	    //
	    // weights instantiation
	    weights_ = std::make_shared< W >( std::shared_ptr< FullyConnectedLayer< AF, W, D > >( this ),
					      fc_layer_size_, prev_layer_size );
	  }
	//
	//
	// Build the activation
	// Get the tensor arrays. In this second loop we gather the information for the activation
	weights_->activate( prev_layer_tensors, activation_func_ );

	
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit(-1);
      }
  };
}
#endif
