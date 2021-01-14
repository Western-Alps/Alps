#ifndef FUNCTION_H
#define FUNCTION_H
#include <memory>
//
//
//
#include "AlpsBaseFunction.h"
//
//
//
namespace MAC
{
  /** \class CostFunction
   *
   * \brief 
   * This class is the head of a composit design to build neural network
   * 
   */
  template< typename Type >
  class CostFunction
  {
  public:
    /** Destructor */
    virtual ~CostFunction(){};


    //
    // Accessors
    //


    //
    // Functions
    //
    // Loss function function
    virtual Type                    L( Type*, Type*, std::size_t )   = 0;
    //
    // Loss function derivative
    virtual std::share_ptr< Type* > dL( Type*, Type*, std::size_t )  = 0;
  };
  /** \class Function
   *
   * \brief 
   * This class is the head of a composit design to build neural network
   * 
   */
  template< typename Type >
  class LeastSquarreEstimate : public Alps::CostFunction< Type >
  {
  public:
    /** Constructor */
    explicite LeastSquarreEstimate(){};
    /** Destructor */
    virtual ~LeastSquarreEstimate(){};
    
    
    //
    // Accessors
    //


    //
    // Functions
    //
    // Loss function function
    virtual Type  L( Type*, Type*, std::size_t );
    //
    // Loss function derivative
    virtual std::share_ptr< Type* > dL( Type*, Type*, std::size_t );
  };
  //
  //
  //
  template< typename Type > Type
  class LeastSquarreEstimate< T >::L( Type*       Optimum,
				      Type*       Target,
				      std::size_t N )
  {
    Type cost = 0;
    for ( std::size_t i = 0 ; i < N ; i++ )
      cost += (Optimum[i] - Target[i] ) * (Optimum[i] - Target[i] );
    //
    //
    return cost;
  }
  //
  //
  //
  template< typename Type > std::share_ptr< Type* >
  class LeastSquarreEstimate< T >::dL( Type*       Optimum,
				       Type*       Target,
				       std::size_t N )
  {
    std::share_ptr< Type* > error ( new Type[N], std::default_delete<  T [] > );
    for ( std::size_t i = 0 ; i < N ; i++ )
      ( error.get() )[i] = (Optimum[i] - Target[i] );
    //
    //
    return error;
  }
}
#endif
