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
#include "MACException.h"
#include "AlpsClimber.h"
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
   * This pattern will be observed by the climbers (observers), e.g. waiting for
   * updates.
   *
   */
  // Observed (subject)
  class Mountain
  {
  public:
    /** Destructor */
    virtual ~Mountain(){};
    

    //
    // Accessors
    //

    
    //
    // functions
    //
    // Attach observers that need to be updated. We use weak pointer to avoid circularity between object.
    virtual void attach( std::shared_ptr< Alps::Climber > )  = 0;
    // Notify the observers for updates
    virtual void notify()                                    = 0;
    // Save the weights at the end of the epoque
    virtual void save_weight_file( const std::size_t ) const = 0;
  };
}
#endif

