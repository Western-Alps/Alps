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
    virtual Type                      L( Type*, Type*, std::size_t )         = 0;
    //
    // Loss function derivative
    virtual std::shared_ptr< Type >  dL( Type*, Type*, Type*, std::size_t )  = 0;
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
    virtual std::shared_ptr< Type >  dL( Type*, Type*, Type*, std::size_t );

      
    private:
      //
      Func name_{Alps::Func::L_LSE};
  };
  //
  //
  //
  template< typename T > T
  LeastSquarreEstimate< T >::L( T*          Optimum,
				T*          Target,
				std::size_t N )
  {
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
  template< typename T > std::shared_ptr< T >
  LeastSquarreEstimate< T >::dL( T*          Optimum,
				    T*          Target,
				    T*          DOptimum,
				    std::size_t N )
  {
    std::shared_ptr< T > error ( new T[N], std::default_delete<  T [] >() );
    for ( std::size_t i = 0 ; i < N ; i++ )
      ( error.get() )[i] = (Optimum[i] - Target[i]) * DOptimum[i];
    //
    //
    // UNITreturn error;
    return nullptr;
  }
}
#endif
