#ifndef ALPSLAYER_H
#define ALPSLAYER_H
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

  /** \class Layer
   *
   * \brief 
   * This class represents the Application Programming Interface for any type 
   * of layer used in the neural network.
   * 
   */
  class Layer
    {
    public:
      /** Destructor */
      virtual ~Layer(){};
      
      //
      // Accessors
      
      // Forward propagation
      virtual void forward()  = 0;
      // Backward propagation
      virtual void backward() = 0;
    };
}
#endif
