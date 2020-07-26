#ifndef ALPSLAYERBASE_H
#define ALPSLAYERBASE_H
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

  /** \class LayerBase
   *
   * \brief 
   * This class represents the Application Programming Interface for the layers of the neural network.
   * 
   */
  class LayerBase
    {
    public:
      /** Destructor */
      virtual ~LayerBase(){};
      
      //
      // Accessors
      
      // Forward propagation
      virtual void forward()  == 0;
      // Backward propagation
      virtual void backward() == 0;
    };
}
#endif
