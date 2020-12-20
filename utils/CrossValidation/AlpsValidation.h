#ifndef ALPSVALIDATION_H
#define ALPSVALIDATION_H
//
//
//
#include <iostream> 
//
// Alps
#include "MACException.h"
//
//
//
/*! \namespace Alps
 *
 * Name space for .
 *
 */
namespace Alps
{
  /*! \class Classifier
   * \brief class representing    
   *
   */
  class Validation
  {
    //
    //
  public:
    // train the calssification engin
    virtual void train() = 0;
    // use the calssification engin
    virtual void use()   = 0;
  };
}
#endif
