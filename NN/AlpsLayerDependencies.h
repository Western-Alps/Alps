#ifndef ALPSLAYERDEPENDENCIESS_H
#define ALPSLAYERDEPENDENCIESS_H
//
//
//
#include <iostream>
#include <memory>
//
#include "MACException.h"
/*! \namespace Alps
 *
 * Name space for Alps.
 *
 */
namespace Alps
{
  //
  // observer forward definition
  class Weights;
  /** \class LayerDependencies
   *
   * \brief 
   * This class represents the Application Programming Interface for the layers dependencies.
   * 
   */
  class LayerDependencies
    {
    protected:
      LayerDependencies();
      
    public:
      /** Destructor */
      virtual ~LayerDependencies(){};
      
      //
      // Accessors
      
      // Update the observers
      virtual void update_weights()                             = 0;
      // Update the observers
      virtual void attach_weights( std::shared_ptr< Weights > ) = 0;
    };
}
#endif
