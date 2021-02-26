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
  template< typename Type, typename Tensor1_Type, typename Tensor2_Type, Alps::Arch Architecture >
  class StochasticGradientDescent : public Alps::Gradient< Tensor1_Type, Tensor2_Type >
  {
  public:
    /** Costructor */
    explicit StochasticGradientDescent(){};
    /** Destructor */
    virtual ~StochasticGradientDescent(){};
    
    
    //
    // Accessors
    //
    // Set the layer sizes
    virtual void            set_parameters( const std::size_t, const std::size_t )      override{};
    
    
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
  template< typename Type >
  class StochasticGradientDescent< Type, Eigen::MatrixXd, Eigen::MatrixXd, Alps::Arch::CPU > : public Alps::Gradient< Eigen::MatrixXd, Eigen::MatrixXd >
    {
     public:
      /** Costructor */
      explicit StochasticGradientDescent();
      /** Destructor */
      virtual ~StochasticGradientDescent(){};

      
      //
      // Accessors
      //
      // Set the layer sizes
      virtual void            set_parameters( const std::size_t, const std::size_t )      override;
      
      
      //
      // Functions
      //
      // Get the type of optimizer
      virtual const Alps::Grad get_optimizer() const
      { return Alps::Grad::SGD;};
      // Add tensor elements
      virtual void            add_tensors( const Eigen::MatrixXd, const Eigen::MatrixXd ) override;
      // Backward propagation
      virtual Eigen::MatrixXd solve()                                                     override;

    private:
      //
      Eigen::MatrixXd delta_;
      //
      Eigen::MatrixXd previous_activation_;
    };
  //
  //
  //
  template< typename Type >
  Alps::StochasticGradientDescent< Type, Eigen::MatrixXd, Eigen::MatrixXd, Alps::Arch::CPU >::StochasticGradientDescent()
  {
  }
  //
  //
  //
  template< typename Type > void
  Alps::StochasticGradientDescent< Type, Eigen::MatrixXd, Eigen::MatrixXd, Alps::Arch::CPU >::set_parameters( const std::size_t Current_size,
													      const std::size_t Prev_size )
  {
    delta_               = Eigen::MatrixXd::Zero( Current_size, 1 );
    previous_activation_ = Eigen::MatrixXd::Zero( Prev_size, 1 );
  }
  //
  //
  //
  template< typename Type > void
  Alps::StochasticGradientDescent< Type, Eigen::MatrixXd, Eigen::MatrixXd, Alps::Arch::CPU >::add_tensors( const Eigen::MatrixXd Delta,
													   const Eigen::MatrixXd Z )
  {
    try
      {
	if ( delta_.rows() != Delta.rows() || previous_activation_.rows() != Z.rows() )
	  throw MAC::MACException( __FILE__, __LINE__,
				   "Missmatched dimensions.",
				   ITK_LOCATION );
	//
	//
	delta_               += Delta;
	previous_activation_ += Z;
      }
    catch( itk::ExceptionObject & err )
      {
     	std::cerr << err << std::endl;
	exit(-1);
      }
  }
  //
  //
  //
  template< typename Type > Eigen::MatrixXd
  Alps::StochasticGradientDescent< Type, Eigen::MatrixXd, Eigen::MatrixXd, Alps::Arch::CPU >::solve()
  {
    return delta_ * previous_activation_.transpose();
  }
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
  template< typename Type, typename Tensor1_Type, typename Tensor2_Type >
  class StochasticGradientDescent< Type, Tensor1_Type, Tensor2_Type, Alps::Arch::CUDA > : public Alps::Gradient< Tensor1_Type, Tensor2_Type >
    {
     public:
      /** Costructor */
      explicit StochasticGradientDescent(){};
      /** Destructor */
      virtual ~StochasticGradientDescent(){};

      
      //
      // Accessors
      //
      // Set the layer sizes
      virtual void            set_parameters( const std::size_t, const std::size_t )      override{};

      
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
