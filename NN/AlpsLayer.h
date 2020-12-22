#ifndef ALPSLAYER_H
#define ALPSLAYER_H
//
//
//
#include <iostream>
#include <vector>
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
      virtual       void             set_next_layer( std::shared_ptr< Alps::Layer > ) = 0;
      virtual const std::vector<int> get_layer_size() const                           = 0;
      // Functions
      // Forward propagation
      virtual       void forward()                                                    = 0;
      // Backward propagation
      virtual       void backward()                                                   = 0;
    };
}
#endif
