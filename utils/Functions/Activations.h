#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
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
  /** \class Activation_tanh
   *
   * \brief 
   * This class is hyperbolic tangent
   * \mathbb{R} \leftrightarrow [-1,1]
   * 
   */
  class Activation_tanh : public Functions
    {
    public:
      /** Constructor. */
      explicit Activation_tanh(){};
      
      /** Destructor */
      virtual ~Activation_tanh(){};

      //
      // activation function
      virtual double f( const double X ){return tanh(X) + 1.e-10;};
      //
      // activation function derivative
      virtual double df( const double X ){return 1. - tanh(X)*tanh(X);};

    private:
      //
      // Loss function function
      virtual double L( const double, const double ){return 0.;};
      //
      // Loss function derivative
      virtual double dL( const double, const double ){return 0.;};
    };
  /** \class Activation_sigmoid
   *
   * \brief 
   * This class is logistic sigmoid 
   * \mathbb{R} \leftrightarrow [0,1]
   * 
   */
  class Activation_sigmoid : public Functions
    {
    public:
      /** Constructor. */
      explicit Activation_sigmoid(){};
      
      /** Destructor */
      virtual ~Activation_sigmoid(){};

      //
      // activation function
      virtual double f( const double X ){return 1. / (1. + exp(X) );};
      //
      // activation function derivative
      virtual double df( const double X ){return f(X)*(1. - f(X));};

    private:
      //
      // Loss function function
      virtual double L( const double, const double ){return 0.;};
      //
      // Loss function derivative
      virtual double dL( const double, const double ){return 0.;};
    };
}
#endif
