#ifndef ALPSWEIGHTS_H
#define ALPSWEIGHTS_H
//
//
//
#include <iostream> 
//
#include "MACException.h"
/*! \namespace Alps
 *
 * Name space for Alps.
 *
 */
namespace Alps
{
  /*! \class Weights
   * \brief class representing the weights container between all the neural networks layers.
   *
   */
  class Weights
  {
  public:
    // Destructor
    virtual ~Weights();


  public:
    //
    // Save the weightd
    virtual void save_weights()  == 0;
    // Save the weightd
    virtual void load_weights()  == 0;
    // Save the weightd
    virtual void update()         == 0;
  };
}
#endif
