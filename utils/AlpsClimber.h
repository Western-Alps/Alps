#ifndef ALPSCLIMBER_H
#define ALPSCLIMBER_H
//
//
//
#include <iostream> 
#include <memory>
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

    //
    virtual ~Climber(){};

    //
    // Accessors


    //
    // functions
    //
    virtual void update() = 0;
  };
}
#endif
