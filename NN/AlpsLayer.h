#ifndef ALPSLAYER_H
#define ALPSLAYER_H
//
//
//
#include <iostream>
#include <vector>
//
#include "MACException.h"
#include "AlpsClimber.h"
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
      //
      // get the layer identification
      virtual const std::size_t              get_layer_id() const                             = 0;
      // get the layer name
      virtual const std::string              get_layer_name() const                           = 0;
      // get number of weights
      virtual const int                      get_number_weights() const                       = 0;
      // get the layer size
      virtual const std::vector<std::size_t> get_layer_size() const                           = 0;
      // attach the next layer
      virtual void                           set_next_layer( std::shared_ptr< Alps::Layer > ) = 0;


      //
      // Functions
      //
      // Add previous layer
      virtual void add_layer( std::shared_ptr< Alps::Layer > ) = 0;
      // Forward propagation
      virtual void forward( std::shared_ptr< Alps::Climber > ) = 0;
      // Backward propagation
      virtual void backward()                                  = 0;
    };
}
#endif
