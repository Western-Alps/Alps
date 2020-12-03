#ifndef ALPSFULLYCONNECTEDLAYER_H
#define ALPSFULLYCONNECTEDLAYER_H
//
//
//
#include <iostream>
#include <memory>
//
#include "MACException.h"
#include "AlpsLayer.h"
#include "AlpsLayerDependencies.h"
#include "AlpsWeights.h"
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
  template< class Container >
    class FullyConnectedLayer : public Layer, public LayerDependencies
  {
    //
    // 
  public:
    /** Constructor. */
    explicit FullyConnectedLayer();
    
    /** Destructor */
    virtual ~FullyConnectedLayer(){};

    //
    // Accessors

    // Forward propagation
    virtual void forward()  override {};
    // Backward propagation
    virtual void backward() override {};
    // Update the weights
    virtual void update_weights() override {};
    // Update the weights
    virtual void attach_weights( std::shared_ptr< Weights > Weights ) override
    { weights_ = Weights; };

  private:
    //
    // private member function
    //

    //
    // Layer owned

    //
    //! Weights is an external dependencies
    std::shared_ptr< Weights > weights_;

    //
    //! this member represents the activation
    //! $a_{i}^{\mu} = \sum_{j}^{N_{\nu}} \omega_{ij}^{\mu} z_{j}^{\nu} + b_{i}^{\mu}$
    Container a_;
    //! $z_{i}^{\mu} = f( a_{i}^{\mu} )$
    Container z_;
    //! Weights $\omega_{ij}$ of the layer

    //
    // Other layers owned

    //
    //! layer in the downward position in the neurla network
    std::shared_ptr< FullyConnectedLayer< Container > > z_down_;
    //! layer in the upward position in the neurla network
    std::shared_ptr< FullyConnectedLayer< Container > > z_up_;
  };
  //
  //
  template< class C >
    FullyConnectedLayer< C >::FullyConnectedLayer()
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
}
#endif
