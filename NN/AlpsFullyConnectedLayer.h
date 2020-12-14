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
#include "AlpsSubjects.h"
#include "AlpsMountain.h"
//#include "AlpsLayerDependencies.h"
//#include "AlpsWeights.h"
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
  class FullyConnectedLayer : public Alps::Layer, public Alps::Mountain
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
    virtual       void             set_next_layer( std::shared_ptr< Alps::Layer > Next ) override
      { next_layer_ = Next;};
    virtual const std::vector<int> get_layer_size() const                                override
      {return fc_layer_size_;}
    //
    // Functions
    // Forward propagation
    virtual       void             forward()                                             override {};
    // Backward propagation
    virtual       void             backward()                                            override {};
    // Attach observers that need to be updated
    virtual       void             attach( std::shared_ptr< Alps::Climber > )            override {};
    // Notify the observers for updates
    virtual       void             notify()                                              override {};

  private:
    //
    // private member function
    //

      
    //
    // layer's name
    std::string                                                layer_name_;
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
    std::tuple< std::shared_ptr< Alps::Climber > /*images*/,
		std::shared_ptr< Alps::Climber > /*weights*/ > climbers_;
  };
  //
  //
  template< typename AF, typename W, int D   >
  FullyConnectedLayer< AF, W, D >::FullyConnectedLayer( const std::string              Layer_name,
							const std::vector<int>         Fc_layer_size,
							std::shared_ptr< Alps::Layer > Prev_layer ):
    layer_name_{Layer_name}, fc_layer_size_{Fc_layer_size}, prev_layer_{Prev_layer}
  {
    try
      {
	//
	//
	
	
	//
	// Create the subjects (images)
	std::shared_ptr< Alps::Subjects< /*ActivationFunction,*/ W, D > >
	  subjects = std::make_shared< Alps::Subjects< /*AF,*/ W, D > >( std::shared_ptr< FullyConnectedLayer< AF, W, D > >( this ) );
	//
	// Create the weights
	std::shared_ptr< W >
	  weights = std::make_shared< W >( std::shared_ptr< FullyConnectedLayer< AF, W, D > >( this ),
					   Fc_layer_size, Prev_layer->get_layer_size() );
	//
	//
	climbers_ = std::make_tuple( subjects, weights );
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit(-1);
      }
  };
}
#endif
