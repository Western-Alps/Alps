#ifndef ALPSCROSSVALIDATION_H
#define ALPSCROSSVALIDATION_H
//
//
//
#include <iostream> 
//
// Alps
#include "MACException.h"
#include "AlpsValidation.h"
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
  temaplate < typename ValidationMethod, typename Mountain >
  class CrossValidation : public Alps::Validation
  {
    //
    //
  public:
    // train the calssification engin
    virtual void train() override {};
    // use the calssification engin
    virtual void use()   override {};;
  };
}
#endif
