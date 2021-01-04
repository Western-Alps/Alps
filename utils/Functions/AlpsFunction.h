#ifndef ALPSFUNCTION_H
#define ALPSFUNCTION_H
//
//
//
#include "MACException.h"
//
//
//
namespace Alps
{
  enum Func { UNDETERMINED = 0,
	      F_TANH       = 1,
	      F_SIGMOID    = 2,
	      F_SSD        = 100 };
  /** \class Function_base
   *
   * \brief 
   * This class is the base class for all the functions.
   * 
   */
  class Function
    {
    public:
      /** Destructor */
      virtual ~Function(){};


      //
      // Accessors
      //
      // Loss function derivative
      virtual Func get_function_name() const = 0;


      //
      // Functions
      //
      // activation function
      virtual double f( const double )           = 0;
      //
      // activation function derivative
      virtual double df( const double )          = 0;
    };
}
#endif
