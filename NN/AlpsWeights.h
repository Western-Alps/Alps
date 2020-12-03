#ifndef ALPSWEIGHTS_H
#define ALPSWEIGHTS_H
//
//
//
#include <iostream> 
#include <memory>
//
#include "AlpsLayerDependencies.h"
#include "MACException.h"
/*! \namespace Alps
 *
 * Name space for Alps.
 *
 */
namespace Alps
{
  /*! \class Weights
   * \brief class representing the weights container used by all the neural networks layers.
   *
   */
  class Weights
  {
  protected:
    Weights(){};
  public:
    // Destructor
    virtual ~Weights(){};


  public:
    //
    // Save the weightd
    virtual void save_weights() const = 0;
    // Save the weightd
    virtual void load_weights()      = 0;
    // Save the weightd
    virtual void update( )            = 0;
  };
}
#endif
