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
    virtual std::shared_ptr< Alps::Mountain >                get_mountain()                            = 0;

    
    //
    // functions
    //
    // Update the information
    virtual void                                             update()                                  = 0;
  };
}
#endif
