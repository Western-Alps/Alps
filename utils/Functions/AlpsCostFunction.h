#ifndef FUNCTION_H
#define FUNCTION_H
//
//
//
#include "MACException.h"
//
//
//
namespace MAC
{
  /** \class Function
   *
   * \brief 
   * This class is the head of a composit design to build neural network
   * 
   */
  class CostFunction
    {
    public:
      /** Destructor */
      virtual ~CostFunction(){};


//     //
//     // Accessors
//     //
//     // Loss function derivative
//     virtual Func get_function_name() const          = 0;
//
//
//     //
//     // Functions
//     //
//     // activation function
//     virtual double f( const double )                = 0;
//     //
//     // activation function derivative
//     virtual double df( const double )               = 0;
//     //
//     // Loss function function
//     virtual double L( const double, const double )  = 0;
//     //
//     // Loss function derivative
//     virtual double dL( const double, const double ) = 0;
    };
}
#endif
