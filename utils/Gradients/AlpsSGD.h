#ifndef ALPSSGD_H
#define ALPSSGD_H
//
//
//
#include <iostream>
#include <memory>
//
//
//
#include "MACException.h"
#include "AlpsGradient.h"
//
//
//
namespace Alps
{
  /** \class Gradient
   *
   * \brief 
   * This class is the stochastic gradient descent class.
   * 
   */
  template< typename Type >
    class SGD : public Alps::Gradient< Type >
    {
     public:
      /** Costructor */
      explicit SGD(){};
      /** Destructor */
      virtual ~SGD(){};

      //
      // Backward propagation
      virtual void solve(){};
    };
}
#endif
