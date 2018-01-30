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
  enum Func { UNDETERMINED = 0,
	      F_TANH       = 1,
	      F_SIGMOID    = 2,
	      F_SSD        = 100 };
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
      virtual __host__ __device__ double f( const double )  = 0;
      //
      // activation function derivative
      virtual __host__ __device__ double df( const double ) = 0;
      //
      // Loss function function
      virtual __host__ __device__ double L( const double, const double )  = 0;
      //
      // Loss function derivative
      virtual __host__ __device__ double dL( const double, const double ) = 0;
      //
      // Loss function derivative
      virtual Func get_function_name() const = 0;
    };
}
#endif
