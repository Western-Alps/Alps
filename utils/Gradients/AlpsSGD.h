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
#include "AlpsLoadDataSet.h"
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
    explicit StochasticGradientDescent() = default;
    /** Destructor */
    virtual ~StochasticGradientDescent() = default;
    
    
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
    virtual void         add_tensors( const Tensor1_Type&, const Tensor1_Type& ) override {};
    // Backward propagation
    virtual Tensor2_Type solve( const bool = false )                           override
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
  class StochasticGradientDescent< Type, std::vector< Type >, std::vector< Type >, Alps::Arch::CPU > :
    public Alps::Gradient< std::vector< Type >, std::vector< Type > >
  {
  public:
    /** Costructor */
    explicit StochasticGradientDescent();
    /** Destructor */
    virtual ~StochasticGradientDescent() = default;
    
      
    //
    // Accessors
    //
    // Set the layer sizes
    virtual void                set_parameters( const std::size_t,
						const std::size_t )        override;
      
      
    //
    // Functions
    //
    // Get the type of optimizer
    virtual const Alps::Grad    get_optimizer() const
    { return Alps::Grad::SGD;};
    // Add tensor elements
    virtual void                add_tensors( const std::vector< Type >&,
					     const std::vector< Type >& )  override;
    // Backward propagation
    virtual std::vector< Type > solve( const bool = false)                 override;

  private:
    //
    // Gradient information
    // mini batch size
    std::size_t             mini_batch_{0};
    // batch represent the current state before update of the weights
    std::size_t             batch_{0};
    // learning rate
    double                  learning_rate_{0.00001};
    //
    // Update of the weights
    // number of weights
    std::size_t             delta_size_{0};
    // delta_ = [W epsilon] * ...
    std::vector< Type >     delta_;
    // activation from the previous layer
    std::vector< Type >     previous_activation_;
  };
  //
  //
  //
  template< typename Type >
  Alps::StochasticGradientDescent< Type, std::vector< Type >, std::vector< Type >, Alps::Arch::CPU >::StochasticGradientDescent()
  {
    //mini_batch_    = static_cast< std::size_t >(Alps::LoadDataSet::instance()->get_data()["mountain"]["strategy"]["mini_batch"]);
    learning_rate_ = static_cast< double >(Alps::LoadDataSet::instance()->get_data()["mountain"]["strategy"]["learning_rate"]);
  }
  //
  //
  //
  template< typename Type > void
  Alps::StochasticGradientDescent< Type, std::vector< Type >, std::vector< Type >, Alps::Arch::CPU >::set_parameters( const std::size_t Current_size,
														      const std::size_t Prev_size )
  {
    delta_size_          = Current_size;
    delta_               = std::vector< Type >( Current_size, 0. );
    //previous_activation_ = nullptr;
  }
  //
  //
  //
  template< typename Type > void
  Alps::StochasticGradientDescent< Type, std::vector< Type >, std::vector< Type >, Alps::Arch::CPU >::add_tensors( const std::vector< Type >& Delta,
														   const std::vector< Type >& Z )
  {
    try
      {
	//
	//
	for ( std::size_t d = 0 ; d < delta_size_ ; d++ )
	  delta_[d] -= learning_rate_ * Delta[d];
	// An additional image, we increase the batch size
	batch_++;
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
  template< typename Type > std::vector< Type >
  Alps::StochasticGradientDescent< Type, std::vector< Type >, std::vector< Type >, Alps::Arch::CPU >::solve( const bool Forced )
  {
    if ( batch_ > mini_batch_ - 1 || Forced )
      {
	batch_ = 1;
	return delta_;
      }
    else
      return std::vector< Type >( delta_size_, 0. );
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
  template< typename Type >
  class StochasticGradientDescent< Type, Eigen::MatrixXd, Eigen::MatrixXd, Alps::Arch::CPU > : public Alps::Gradient< Eigen::MatrixXd, Eigen::MatrixXd >
  {
  public:
    /** Costructor */
    explicit StochasticGradientDescent();
    /** Destructor */
    virtual ~StochasticGradientDescent() = default;

      
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
    virtual void            add_tensors( const Eigen::MatrixXd&, const Eigen::MatrixXd& ) override;
    // Backward propagation
    virtual Eigen::MatrixXd solve( const bool = false)                                  override;

  private:
    //
    // Gradient information
    // mini batch size
    std::size_t     mini_batch_{0};
    // batch represent the current state before update of the weights
    std::size_t     batch_{0};
    // learning rate
    double          learning_rate_{0.00001};
    //
    // Update of the weights
    // delta_ = [W epsilon] * ...
    Eigen::MatrixXd delta_;
    // activation from the previous layer
    Eigen::MatrixXd previous_activation_;
  };
  //
  //
  //
  template< typename Type >
  Alps::StochasticGradientDescent< Type, Eigen::MatrixXd, Eigen::MatrixXd, Alps::Arch::CPU >::StochasticGradientDescent()
  {
    //mini_batch_    = static_cast< std::size_t >(Alps::LoadDataSet::instance()->get_data()["mountain"]["strategy"]["mini_batch"]);
    learning_rate_ = static_cast< double >(Alps::LoadDataSet::instance()->get_data()["mountain"]["strategy"]["learning_rate"]);
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
  Alps::StochasticGradientDescent< Type, Eigen::MatrixXd, Eigen::MatrixXd, Alps::Arch::CPU >::add_tensors( const Eigen::MatrixXd& Delta,
													   const Eigen::MatrixXd& Z )
  {
    try
      {
	if ( delta_.rows() != Delta.rows() || previous_activation_.rows() != Z.rows() )
	  {
	    std::string mess = "Dimension mismatch. Parameters are set to: \n ["
	      + std::to_string( delta_.rows() ) + "x"
	      + std::to_string( previous_activation_.rows() ) + "]. \n The tensor added have" +
	      + " the dimension: \n [" + std::to_string( Delta.rows() ) + "x"
	      + std::to_string( Z.rows() ) + "].";
	    throw MAC::MACException( __FILE__, __LINE__,
				     mess.c_str(),
				     ITK_LOCATION );
	  }
	//
	//
	delta_               += Delta;
	previous_activation_ += Z;
	// An additional image, we increase the batch size
	batch_++;
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
  Alps::StochasticGradientDescent< Type, Eigen::MatrixXd, Eigen::MatrixXd, Alps::Arch::CPU >::solve( const bool Forced )
  {
    if ( batch_ > mini_batch_  )
      {
	batch_ = 1;
	return - learning_rate_ * delta_ * previous_activation_.transpose();
      }
    else
      return Eigen::MatrixXd::Zero( delta_.rows(), previous_activation_.rows() );
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
    virtual ~StochasticGradientDescent() = default;

      
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
    virtual void         add_tensors( const Tensor1_Type&, const Tensor1_Type& ) override {};
    // Backward propagation
    virtual Tensor2_Type solve( const bool = false )                                     override
    { return Tensor2_Type();};
  };
}
#endif
