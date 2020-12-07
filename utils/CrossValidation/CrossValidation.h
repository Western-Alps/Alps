#ifndef CROSSVALIDATION_H
#define CROSSVALIDATION_H
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
/*! \namespace CrossValidation
 *
 * Name space for .
 *
 */
namespace MAC
{
  /*! \class Classifier
   * \brief class representing    
   *
   */
  class CrossValidation
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
