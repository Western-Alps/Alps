#ifndef FUNCTIONS_H
#define FUNCTIONS_H
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
  /** \class Functions
   *
   * \brief 
   * This class is the head of a composit design to build neural network
   * 
   */
  class Functions
    {
 
      //private:
      //
      //explicit Subject( const int, const int );

    public:
      /** Constructor. */
      //Functions(){};

      /** Destructor */
      //virtual ~Functions(){};

      //
      // activation function
      virtual double f( const double )  = 0;
      //
      // activation function derivative
      virtual double df( const double ) = 0;
      //
      // Loss function function
      virtual double L( const double, const double )  = 0;
      //
      // Loss function derivative
      virtual double dL( const double, const double ) = 0;
    };
}
#endif
