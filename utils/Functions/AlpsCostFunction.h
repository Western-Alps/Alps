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
    virtual Type                   L( const std::vector< Type >&,
				      const std::vector< Type >&,
				      const std::size_t )          = 0;
    //
    // Loss function derivative
    virtual std::vector< Type >  dL( const std::vector< Type >&,
				     const std::vector< Type >&,
				     const std::vector< Type >&,
				     const std::size_t )            = 0;
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
    virtual Func                    get_function_name() const    override
    {return name_;};
 

    //
    // Functions
    //
    // Loss function function
    virtual Type                   L( const std::vector< Type >&,
				      const std::vector< Type >&,
				      const std::size_t )              override;
    //
    // Loss function derivative
    virtual std::vector< Type >  dL( const std::vector< Type >&,
				     const std::vector< Type >&,
				     const std::vector< Type >&,
				     const std::size_t )           override;

      
    private:
      //
      Func name_{Alps::Func::L_LSE};
  };
  //
  //
  //
  template< typename T > T
  LeastSquarreEstimate< T >::L( const std::vector< T >& Optimum,
				const std::vector< T >& Target,
				const std::size_t       N )
  {
    //
    // In this function the cost is calculated at the level of the image
    // I number of images 
    // E = (1/2I) Sum_{i} || z - t ||^{2} = (1/2I) Sum_{i} E_{i}
    // cost = E_{i}
    //
    T cost = 0;
    for ( std::size_t i = 0 ; i < N ; i++ )
      {
	cost += (Optimum[i] - Target[i]) * (Optimum[i] - Target[i]);
//	std::cout
//	  << "Optimum["<<i<<"] " <<Optimum[i]
//	  << "  Target[i] " <<Target[i]
//	  << std::endl;
      }
    //
    //
    return cost;
  }
  //
  //
  //
  template< typename T > std::vector< T >
  LeastSquarreEstimate< T >::dL( const std::vector< T >& Optimum,
				 const std::vector< T >& Target,
				 const std::vector< T >& DOptimum,
				 const std::size_t       N )
  {
    std::vector< T > error ( N, 0. );
    for ( std::size_t i = 0 ; i < N ; i++ )
      error[i] = (Optimum[i] - Target[i]) * DOptimum[i];
    //
    //
    return error;
  }
}
#endif
