#ifndef GRADIENT_H
#define GRADIENT_H
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
//
//
//
namespace MAC
{
  /** \class Gradient
   *
   * \brief 
   * This class is 
   * 
   */
  class Gradient
    {
 
    protected:
      /** Constructor. */
      Gradient(){};
      //
      //explicit Subject( const int, const int );

    public:
      /** Destructor */
      virtual ~Gradient(){};

      //
      // Backward propagation
      virtual void backward(){};
    };
}
#endif
