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
  enum class Grad
    {
     UNKNOWN   = -1,
     // 
     SGD       = 1, // Stochatic Gradient Descent (SGD)
     MOMENTUM  = 2, // Momentum based Gradient Descent
     AdaGrad   = 3, // Adaptive Gradient Descent
     AdaDelta  = 4, // AdaDelta
     Adam      = 5  // It is not an acronym. The algorithm is called Adam.
    }; 
  /** \class Gradient_base
   *
   * \brief 
   * This class is the base class for all the gradient algorithms.
   *
   */
  class Gradient_base
  {
  public:
    /** Destructor */
    virtual ~Gradient_base() = default;

    //
    // Functions
    //
    // Get the type of optimizer
    virtual const Alps::Grad get_optimizer() const = 0;
  };
  /** \class SGD
   *
   * \brief 
   * This class is the Stochatic Gradient Descent (SGD). This is a simplified class
   * to create a strategy.
   *
   */
  class SGD : public Gradient_base
  {
  public:
    /** Constructor */
    explicit SGD() = default;
    /** Destructor */
    virtual ~SGD() = default;


    //
    // Functions
    //
    // Get the type of optimizer
    virtual const Alps::Grad get_optimizer() const
    { return Alps::Grad::SGD;};
  };
  /** \class Momentum
   *
   * \brief 
   * This class is the Momentum based Gradient Descent. This is a simplified class
   * to create a strategy.
   *
   */
  class Momentum : public Gradient_base
  {
  public:
    /** Constructor */
    explicit Momentum() = default;
    /** Destructor */
    virtual ~Momentum() = default;


    //
    // Functions
    //
    // Get the type of optimizer
    virtual const Alps::Grad get_optimizer() const
    { return Alps::Grad::MOMENTUM;};
  };
  /** \class ADAGRAD
   *
   * \brief 
   * This class is the Adaptive Gradient Descent. This is a simplified class
   * to create a strategy.
   *
   */
  class AdaGrad : public Gradient_base
  {
  public:
    /** Constructor */
    explicit AdaGrad() = default;
    /** Destructor */
    virtual ~AdaGrad() = default;

      
    //
    // Functions
    //
    // Get the type of optimizer
    virtual const Alps::Grad get_optimizer() const
    { return Alps::Grad::AdaGrad;};
  };
  /** \class ADADELTA
   *
   * \brief 
   * This class is the Adaptive delta. This is a simplified class
   * to create a strategy.
   *
   */
  class AdaDelta : public Gradient_base
  {
  public:
    /** Constructor */
    explicit AdaDelta() = default;
    /** Destructor */
    virtual ~AdaDelta() = default;

      
    //
    // Functions
    //
    // Get the type of optimizer
    virtual const Alps::Grad get_optimizer() const
    { return Alps::Grad::AdaDelta;};
  };
  /** \class Adam
   *
   * \brief 
   * This class is the Adam optimizer. This is a simplified class
   * to create a strategy.
   *
   */
  class Adam : public Gradient_base
  {
  public:
    /** Constructor */
    explicit Adam() = default;
    /** Destructor */
    virtual ~Adam() = default;

      
    //
    // Functions
    //
    // Get the type of optimizer
    virtual const Alps::Grad get_optimizer() const
    { return Alps::Grad::Adam;};
  };
  /** \class Gradient
   *
   * \brief 
   * This class is the base class for the gradient algorithms forcing specific operations.
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
  class Gradient : public Gradient_base
  {
  public:
    /** Destructor */
    virtual ~Gradient() = default;

    
    //
    // Accessors
    //
    // Set the layer sizes
    virtual void            set_parameters( const std::size_t, const std::size_t )   = 0;
    //
    // Reset the layer sizes
    virtual void            reset_parameters()                                       = 0;
      
 
    //
    // Functions
    //
    // Get the type of optimizer
    virtual const Alps::Grad get_optimizer() const                                   = 0;
    // Add tensor elements
    virtual void             add_tensors( const Tensor1_Type&, const Tensor1_Type& ) = 0;
    // Backward propagation
    virtual Tensor2_Type     solve( const bool = false)                              = 0;
  };
}
#endif
