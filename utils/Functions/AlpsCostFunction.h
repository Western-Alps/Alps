#ifndef COSTFUNCTION_H
#define COSTFUNCTION_H
#include <memory>
//
//
//
#include "AlpsBaseFunction.h"
//
//
//
namespace Alps
{
  /** \class CostFunction
   *
   * \brief 
   * This class is the head of a composit design to build neural network
   * 
   */
  template< typename Type>
  class CostFunction : public Alps::BaseFunction
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
    virtual Type                      L( Type*, Type*, std::size_t )  = 0;
    //
    // Loss function derivative
    virtual std::shared_ptr< Type* > dL( Type*, Type*, std::size_t )  = 0;
  };
  /** \class Function
   *
   * \brief 
   * This class is the head of a composit design to build neural network
   * 
   */
  template< typename Type, typename Activation  >
  class LeastSquarreEstimate : public Alps::CostFunction< Type >
  {
  public:
    /** Constructor */
    explicit LeastSquarreEstimate(){};
    /** Destructor */
    virtual ~LeastSquarreEstimate(){};
    
    
    //
    // Accessors
    //
    // get function name 
    virtual Func                    get_function_name() const {return name_;};
 

    //
    // Functions
    //
    // Loss function function
    virtual Type                      L( Type*, Type*, std::size_t );
    //
    // Loss function derivative
    virtual std::shared_ptr< Type* > dL( Type*, Type*, std::size_t );

      
    private:
      //
      Func name_{Alps::Func::L_LSE};
  };
  //
  //
  //
  template< typename T, typename A > T
  LeastSquarreEstimate< T, A >::L( T*          Optimum,
				   T*          Target,
				   std::size_t N )
  {
    A activation;
    T cost = 0;
    for ( std::size_t i = 0 ; i < N ; i++ )
      cost += (Optimum[i] - Target[i]) * (Optimum[i] - Target[i]);

    //
    //
    return cost;
  }
  //
  //
  //
  template< typename T, typename A > std::shared_ptr< T* >
  LeastSquarreEstimate< T, A >::dL( T*          Optimum,
				    T*          Target,
				    std::size_t N )
  {
// UNIT    std::shared_ptr< T > error ( new T[N], std::default_delete<  T [] >() );
// UNIT    for ( std::size_t i = 0 ; i < N ; i++ )
// UNIT      ( error.get() )[i] = 2 * (Optimum[i] - Target[i]) * activation.df( Optimum[i] );
    //
    //
    // UNITreturn error;
    return nullptr;
  }
}
#endif
