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
#include "AlpsBaseFunction.h"
//
//
//
namespace Alps
{
  /** \class Function_base
   *
   * \brief 
   * This class is the base class for all the functions.
   * 
   */
  template< typename Type >
  class Function : public Alps::BaseFunction
    {
    public:
      /** Destructor */
      virtual ~Function(){};


      //
      // Accessors
      //


      //
      // Functions
      //
      // activation function
      virtual Type f( const Type )           = 0;
      //
      // activation function derivative
      virtual Type df( const Type )          = 0;
    };
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
      Func name_{Alps::Func::F_TANH};
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
      virtual Type f( const Type X ){return 1. / (1. + exp(-X) ) + 1.e-10;};
      // activation function derivative
      virtual Type df( const Type X ){return f(X) * (1. - f(X));};

      
    private:
      //
      Func name_{Alps::Func::F_SIGMOID};
    };
  /** \class Activation_ReLU
   *
   * \brief 
   * This class is Rectified Linear Unit 
   * f(x) = x if x > 0 otherwise 0
   * 
   */
  template< typename Type >
  class Activation_ReLU : public Alps::Function<Type>
    {
    public:
      /** Constructor. */
      explicit Activation_ReLU(){};
      
      /** Destructor */
      virtual ~Activation_ReLU(){};


      //
      // Accessors
      //
      // get function name 
      virtual Func get_function_name() const {return name_;};


      //
      // Functions
      //
      // activation function
      virtual Type f( const Type X ){return ( X > 0 ? X : 0.) + 1.e-10;};
      // activation function derivative
      virtual Type df( const Type X ){return ( X > 0 ? 1. : 0.);};

      
    private:
      //
      Func name_{Alps::Func::F_RELU};
    };
  /** \class Activation_linear
   *
   * \brief 
   * This class is linear Unit 
   * f(x) = x 
   * 
   */
  template< typename Type >
  class Activation_linear : public Alps::Function<Type>
    {
    public:
      /** Constructor. */
      explicit Activation_linear(){};
      
      /** Destructor */
      virtual ~Activation_linear(){};


      //
      // Accessors
      //
      // get function name 
      virtual Func get_function_name() const {return name_;};


      //
      // Functions
      //
      // activation function
      virtual Type f( const Type X ){return X;};
      // activation function derivative
      virtual Type df( const Type X ){return 1.;};

      
    private:
      //
      Func name_{Alps::Func::F_LINEAR};
    };
}
#endif
