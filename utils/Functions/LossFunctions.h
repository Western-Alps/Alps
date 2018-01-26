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
      explicit SSD(){};
      
      /** Destructor */
      virtual ~SSD(){};

      //
      // Loss function function
      virtual double L( const double Traget, const double X){return (X - Target)*(X - Target)};
      //
      // Loss function derivative
      virtual double dL( const double Target, const double X){return X - Target;};

    private:
      //
      // activation function
      virtual double f( const double ){return 0.;};
      //
      // activation function derivative
      virtual double df( const double ){return 0.;};

    };
}
#endif
