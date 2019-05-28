#ifndef SGD_H
#define SGD_H
//
//
//
#include <iostream>
#include <fstream>
#include <math.h>  
#include <map>
#include <memory>
#include <list>
//
// CUDA
//
#include <cuda_runtime.h>
//
//
//
#include "MACException.h"
#include "Gradient.h"
//
//
//
namespace MAC
{
  /** \class SGD
   *
   * \brief 
   * This class is a stochastic gradient descent class.
   * 
   */
  class SGD : public Gradient
    {
 
    public:
      /** Constructor. */
      explicit SGD(){};

      /** Destructor */
      virtual ~SGD(){};

      //
      // Backward propagation
      virtual void backward(){};
    };
}
#endif
