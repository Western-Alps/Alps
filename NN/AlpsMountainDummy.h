#ifndef ALPMOUNTAINDUMMY_H
#define ALPMOUNTAINDUMMY_H
//
//
//
#include <iostream> 
#include <memory>
//
// 
//
#include "AlpsClimber.h"
#include "AlpsMountain.h"
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
   * \brief class MountainDummy is a dummy instanciation of the interface for tests.
   * 
   * This pattern will be observed by the climbers (observers) for instant 
   * updates.
   *
   */
  // Observed (subject)
  class MountainDummy : public Alps::Mountain
  {
  public:
    
    //
    MountainDummy(){};
    //
    virtual ~MountainDummy(){};

    //
    // Accessors


    //
    // functions
    //
    // Attach observers that need to be updated
    virtual void attach( std::shared_ptr< Alps::Climber > ) override {};
    // Notify the observers for updates
    virtual void notify() override {};
  };
}
#endif

