#ifndef LOSSFUNCTIONS_H
#define LOSSFUNCTIONS_H
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
#include "Functions.h"
//
//
//
namespace MAC
{
  /** \class SSD
   *
   * \brief 
   * This class is sum of squared differences (SSD)
   * 
   */
  class SSD : public Functions
    {
    public:
      /** Constructor. */
      __host__ __device__ explicit SSD(){};
      
      /** Destructor */
      __host__ __device__ virtual ~SSD(){};

      //
      // Loss function function
      virtual __host__ __device__ double L( const double Traget, const double X){return (X - Target)*(X - Target)};
      //
      // Loss function derivative
      virtual __host__ __device__ double dL( const double Target, const double X){return X - Target;};
      //
      // get function name 
      virtual Func get_function_name() const {return name_;};

    private:
      //
      // activation function
      virtual __host__ __device__ double f( const double ){return 0.;};
      //
      // activation function derivative
      virtual __host__ __device__ double df( const double ){return 0.;};

      //
      //
      Func name_{F_SSD};
    };
}
#endif
