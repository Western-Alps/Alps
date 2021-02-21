#ifndef ALPSSGD_H
#define ALPSSGD_H
//
//
//
#include <iostream>
#include <memory>
//
//
//
#include "MACException.h"
#include "AlpsGradient.h"
//
//
//
namespace Alps
{
  /** \class SGD
   *
   * \brief 
   * This class is the stochastic gradient descent (SGD) class.
   * 
   * - MiniBatch: the size of the mini-bach
   *   * MiniBatch =  1 -- stochastic gradient descent
   *   * MiniBatch =  n -- batch of size n images, n < N the total 
   *                       number of images
   *   * MiniBatch = -1 -- the model uses all the images
   * 
   */
  template< typename Tensor1_Type,
	    typename Tensor2_Type,
	    Alps::Arch Architecture >
  class StochasticGradientDescent : public Alps::Gradient< Tensor1_Type, Tensor2_Type >, public Alps::SGD
    {
     public:
      /** Costructor */
      explicit StochasticGradientDescent(){};
      /** Destructor */
      virtual ~StochasticGradientDescent(){};

      //
      // Functions
      //
      // Get the type of optimizer
      virtual const Alps::Grad get_optimizer() const
      { return Alps::Grad::SGD;};
      // Add tensor elements
      virtual void         add_tensors( const Tensor1_Type, const Tensor1_Type ) override {};
      // Backward propagation
      virtual Tensor2_Type solve()                                               override
      { return Tensor2_Type();};
    };
  /** \class StochasticGradientDescent
   *
   * \brief 
   * This class is the stochastic gradient descent (SGD) class.
   * 
   * - MiniBatch: the size of the mini-bach
   *   * MiniBatch =  1 -- stochastic gradient descent
   *   * MiniBatch =  n -- batch of size n images, n < N the total 
   *                       number of images
   *   * MiniBatch = -1 -- the model uses all the images
   * 
   */
  template< >
  class StochasticGradientDescent< Eigen::MatrixXd, Eigen::MatrixXd, Alps::Arch::CPU > : public Alps::Gradient< Eigen::MatrixXd, Eigen::MatrixXd >, public Alps::SGD
    {
     public:
      /** Costructor */
      explicit StochasticGradientDescent(){};
      /** Destructor */
      virtual ~StochasticGradientDescent(){};

      //
      // Functions
      //
      // Get the type of optimizer
      virtual const Alps::Grad get_optimizer() const
      { return Alps::Grad::SGD;};
      // Add tensor elements
      virtual void            add_tensors( const Eigen::MatrixXd, const Eigen::MatrixXd ) override {};
      // Backward propagation
      virtual Eigen::MatrixXd solve()                                                     override
      {return Eigen::MatrixXd::Zero(1,1);};
    };
  /** \class StochasticGradientDescent
   *
   * \brief 
   * This class is the stochastic gradient descent (SGD) class.
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
    class StochasticGradientDescent< Tensor1_Type, Tensor2_Type, Alps::Arch::CUDA > : public Alps::Gradient< Tensor1_Type, Tensor2_Type >, public Alps::SGD
    {
     public:
      /** Costructor */
      explicit StochasticGradientDescent(){};
      /** Destructor */
      virtual ~StochasticGradientDescent(){};

      //
      // Functions
      //
      // Get the type of optimizer
      virtual const Alps::Grad get_optimizer() const
      { return Alps::Grad::SGD;};
      // Add tensor elements
      virtual void         add_tensors( const Tensor1_Type, const Tensor1_Type ) override {};
      // Backward propagation
      virtual Tensor2_Type solve()                                               override
      { return Tensor2_Type();};
    };
}
#endif
