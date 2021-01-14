#ifndef ALPSACTIVATIONS_H
#define ALPSACTIVATIONS_H
//
//
//
#include <iostream>
#include <math.h>  
//
//
//
#include "MACException.h"
#include "AlpsFunction.h"
//
//
//
namespace Alps
{
  /** \class Activation_tanh
   *
   * \brief 
   * This class is hyperbolic tangent
   * \mathbb{R} \leftrightarrow [-1,1]
   * 
   */
  template< typename Type >
  class Activation_tanh : public Alps::Function< Type >
    {
    public:
      /** Constructor. */
      explicit Activation_tanh(){};
      
      /** Destructor */
      virtual ~Activation_tanh(){};


      //
      // Accessors
      //
      // get function name 
      virtual Func get_function_name() const {return name_;};


      //
      // Functions
      //
      // activation function
      virtual Type f( const Type X ){return tanh(X) + 1.e-10;};
      //
      // activation function derivative
      virtual Type df( const Type X ){return 1. - tanh(X)*tanh(X);};

      
    private:
      //
      Func name_{F_TANH};
    };
  /** \class Activation_sigmoid
   *
   * \brief 
   * This class is logistic sigmoid 
   * \mathbb{R} \leftrightarrow [0,1]
   * 
   */
  template< typename Type >
  class Activation_sigmoid : public Alps::Function<Type>
    {
    public:
      /** Constructor. */
      explicit Activation_sigmoid(){};
      
      /** Destructor */
      virtual ~Activation_sigmoid(){};


      //
      // Accessors
      //
      // get function name 
      virtual Func get_function_name() const {return name_;};


      //
      // Functions
      //
      // activation function
      virtual Type f( const Type X ){return 1. / (1. + exp(X) );};
      //
      // activation function derivative
      virtual Type df( const Type X ){return f(X)*(1. - f(X));};

      
    private:
      //
      Func name_{F_SIGMOID};
    };
}
#endif
