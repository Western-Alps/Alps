#ifndef ALPSGRADIENT_H
#define ALPSGRADIENT_H
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
namespace Alps
{
  /** \class Gradient
   *
   * \brief 
   * This class is the base class for the gradient algorithms.
   * 
   */
  template< typename Type >
  class Gradient
    {
     public:
      /** Destructor */
      virtual ~Gradient(){};

      //
      // Backward propagation
      virtual void solve() = 0;
    };
}
#endif
