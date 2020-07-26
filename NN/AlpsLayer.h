#ifndef ALPSLAYER_H
#define ALPSLAYER_H
//
//
//
#include <iostream>
#include <memory>
//
#include "MACException.h"
#include "AlpsLayerBase.h"
//
//
//
namespace Alps
{
  /** \class Layer
   *
   * \brief 
   * Layer object represents the basic layer element for any type of neural network.
   * 
   */
  template< class Container, class Weights >
    class Layer : public LayerBase
  {
    //
    // 
  public:
    /** Constructor. */
    explicit Layer(){};
    
    /** Destructor */
    virtual ~Layer(){};

    //
    // Accessors

    // Forward propagation
    virtual void forward(){};
    // Backward propagation
    virtual void backward(){};

  private:
    //
    // private member function
    //

    //
    // Layer owned

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
    std::shared_ptr< Layer< Container, Weights > > z_down_;
    //! layer in the upward position in the neurla network
    std::shared_ptr< Layer< Container, Weights > > z_up_;
  };
  //
  //
  template< class C, class W >
    Layer< C, W >::Layer()
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
