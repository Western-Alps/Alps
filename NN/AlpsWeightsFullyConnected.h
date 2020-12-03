#ifndef ALPSWEIGHTSFULLYCONNECTED_H
#define ALPSWEIGHTSFULLYCONNECTED_H
//
//
//
#include <iostream> 
#include <memory>
//
#include "AlpsFullyConnectedLayer.h"
#include "AlpsWeights.h"
#include "MACException.h"
/*! \namespace Alps
 *
 * Name space for Alps.
 *
 */
namespace Alps
{
  // Forward declaration of Conatainer
  
  /*! \class WeightsFullyConnected
   * \brief class representing the weights container between all the neural networks layers.
   *
   */
  template< class Container >
  class WeightsFullyConnected : public Weights
  {
  public:
    // Costructor
    WeightsFullyConnected( std::shared_ptr< FullyConnectedLayer<Container> > );
    // Destructor
    virtual ~WeightsFullyConnected(){};


  public:
    //
    // Save the weights
    virtual void save_weights() const override {};
    // Save the weights
    virtual void load_weights()      override {};
    // Save the weights
    virtual void update()             override {};

  private:
    std::shared_ptr< FullyConnectedLayer<Container> > fully_connected_layer_;
  };

  //
  // Constructor
  temaplte< C >
  Alps::WeightsFullyConnected<C>::WeightsFullyConnected( std::shared_ptr< FullyConnectedLayer<C> > FCLayer )
    {
      fully_connected_layer_ = FCLayer;
      fully_connected_layer_->attach_weights( std::shared_ptr< Weights >(this) );
    }
}
#endif
