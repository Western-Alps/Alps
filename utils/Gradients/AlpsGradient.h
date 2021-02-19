#ifndef ALPSGRADIENT_H
#define ALPSGRADIENT_H
//
//
//
#include <iostream>
#include <memory>
//
//
//
#include "MACException.h"
#include "AlpsWeights.h"
//
//
//
namespace Alps
{
  /** \class Gradient
   *
   * \brief 
   * This class is the base class for the gradient algorithms.
   *
   * - MiniBatch: the size of the mini-bach
   *   * MiniBatch =  1 -- stochastic gradient descent
   *   * MiniBatch =  n -- batch of size n images, n < N the total 
   *                       number of images
   *   * MiniBatch = -1 -- the model uses all the images
   * 
   */
  template< typename Tensor1_Type,
	    typename Tensor2_Type >
  class Gradient
    {
     public:
      /** Destructor */
      virtual ~Gradient(){};

      //
      // Functions
      //
      // Add tensor elements
      virtual void         add_tensors( const Tensor1_Type, const Tensor1_Type ) = 0;
      // Backward propagation
      virtual Tensor2_Type solve()                                               = 0;
    };
}
#endif
