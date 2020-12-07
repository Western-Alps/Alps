#ifndef ALPMOUNTAIN_H
#define ALPMOUNTAIN_H
//
//
//
#include <iostream> 
#include <memory>
//
// 
//
#include "AlpsClimber.h"
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
  /*! \class Mountain
   *
   * \brief class Mountain (subject) is a pur abstraction (interface) for 
   * behavioral observation pattern.
   * 
   * This pattern will be observed by the climbers (observers) for instant 
   * updates.
   *
   */
  // Observed (subject)
  class Mountain
  {
  public:
    
    //
    virtual ~Mountain(){};

    //
    // Accessors


    //
    // functions
    //
    // Attach observers that need to be updated
    virtual void attach( std::shared_ptr< Alps::Climber > ) = 0;
    // Notify the observers for updates
    virtual void notify() = 0;
  };
}
#endif

