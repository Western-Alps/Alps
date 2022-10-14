#ifndef ALPSCLIMBER_H
#define ALPSCLIMBER_H
//
//
//
#include <iostream> 
#include <memory>
#include <vector>
//
// 
//
#include "MACException.h"
//
//
//
/*! \namespace Alps
 *
 * Name space Alps.
 *
 */
namespace Alps
{
  /*! \class Climber
   *
   * \brief class Climber (observer) is a pur abstraction (interface) for 
   * behavioral observation pattern.
   * 
   * This pattern will observe the main mountains (subjects from the pattern) 
   * for instant updates.
   *
   */
  // Observed (subject)
  class Mountain;
  // Observer
  class Climber
  {
  public:
    /* Destructor */
    virtual ~Climber(){};

    
    //
    // Accessors
    //
    // Get the observed mountain
    virtual std::shared_ptr< Alps::Mountain > get_mountain()     = 0;
    // Get energy
    virtual const double                      get_energy() const = 0;
    // Get epoque
    virtual const std::size_t                 get_epoque() const = 0;

    
    //
    // functions
    //
    // Update the information
    virtual void                              update()           = 0;
    // Visualization of the processed image
    virtual void                              visualization()    = 0;
  };
}
#endif
