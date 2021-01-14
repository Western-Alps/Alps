#ifndef ALPSFUNCTION_H
#define ALPSFUNCTION_H
//
//
//
#include "AlpsBaseFunction.h"
//
//
//
namespace Alps
{
  /** \class Function_base
   *
   * \brief 
   * This class is the base class for all the functions.
   * 
   */
  template< typename Type >
  class Function : public Alps::BaseFunction
    {
    public:
      /** Destructor */
      virtual ~Function(){};


      //
      // Accessors
      //


      //
      // Functions
      //
      // activation function
      virtual Type f( const Type )           = 0;
      //
      // activation function derivative
      virtual Type df( const Type )          = 0;
    };
}
#endif
