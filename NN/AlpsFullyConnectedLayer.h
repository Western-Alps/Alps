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
				  const std::vector<int>,
				  std::shared_ptr< Alps::Layer > );
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
    std::string                                                layer_name_{"__Fully_connected_layer__"};
      
    //
    // number of fully connected layers
    std::vector<int>                                           fc_layer_size_;

    //
    // Previous  layers information
    std::shared_ptr< Alps::Layer >                             prev_layer_;
    // Next layers information
    std::shared_ptr< Alps::Layer >                             next_layer_;
    //
    // Observers
    // Observers containers
    std::shared_ptr< Alps::Climber >                           weights_{nullptr};
  };
  //
  //
  //
  template< typename AF, typename W, int D   >
  FullyConnectedLayer< AF, W, D >::FullyConnectedLayer( const std::string              Layer_name,
							const std::vector<int>         Fc_layer_size,
							std::shared_ptr< Alps::Layer > Prev_layer ):
    layer_name_{Layer_name}, fc_layer_size_{Fc_layer_size}, prev_layer_{Prev_layer}
  {
    try
      {}
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
	// If the weights were not initialized yet
	if ( !weights_ )
	  // If no layer is attached to the instanciation of this layer, we select the __input_layer__.
	  weights_ = std::make_shared< W >( std::shared_ptr< FullyConnectedLayer< AF, W, D > >( this ),
					    fc_layer_size_,
					    (prev_layer_ ? prev_layer_->get_layer_size() : subject->get_layer_size() ) );
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit(-1);
      }
  };
}
#endif
