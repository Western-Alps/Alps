#ifndef ALPSADAM_H
#define ALPSADAM_H
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
  /** \class Adam
   *
   * \brief 
   * This class is the Adam class.
   * 
   * - MiniBatch: the size of the mini-bach
   *   * MiniBatch =  1 -- Adam
   *   * MiniBatch =  n -- batch of size n images, n < N the total 
   *                       number of images
   *   * MiniBatch = -1 -- the model uses all the images
   * 
   */
  template< typename Type, typename Tensor1_Type, typename Tensor2_Type, Alps::Arch Architecture >
  class AdamGradient : public Alps::Gradient< Tensor1_Type, Tensor2_Type >
  {
  public:
    /** Costructor */
    explicit AdamGradient() = default;
    /** Destructor */
    virtual ~AdamGradient() = default;
    
    
    //
    // Accessors
    //
    // Set the layer sizes
    virtual void            set_parameters( const std::size_t, const std::size_t )    override{};
    // Set the layer sizes
    virtual void            reset_parameters()                                        override{};
    
    
    //
    // Functions
    //
    // Get the type of optimizer
    virtual const Alps::Grad get_optimizer() const
    { return Alps::Grad::Adam;};
    // Add tensor elements
    virtual void             add_tensors( const Tensor1_Type&, const Tensor1_Type& ) override {};
    // Backward propagation
    virtual Tensor2_Type     solve( const bool = false )                             override
    { return Tensor2_Type();};
  };
  /** \class AdamGradient
   *
   * \brief 
   * This class is the Adam class.
   * 
   * - MiniBatch: the size of the mini-bach
   *   * MiniBatch =  1 -- Adam
   *   * MiniBatch =  n -- batch of size n images, n < N the total 
   *                       number of images
   *   * MiniBatch = -1 -- the model uses all the images
   * 
   */
  template< typename Type >
  class AdamGradient< Type, std::vector< Type >, std::vector< Type >, Alps::Arch::CPU > :
    public Alps::Gradient< std::vector< Type >, std::vector< Type > >
  {
  public:
    /** Costructor */
    explicit AdamGradient();
    /** Destructor */
    virtual ~AdamGradient() = default;
    
      
    //
    // Accessors
    //
    // Set the layer sizes
    virtual void                set_parameters( const std::size_t,
						const std::size_t )        override;
    // Reset the layer sizes
    virtual void                reset_parameters()                         override;
      
      
    //
    // Functions
    //
    // Get the type of optimizer
    virtual const Alps::Grad    get_optimizer() const
    { return Alps::Grad::Adam;};
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
    // cumule of squared gradient
    std::vector< Type >     cumul_squared_gradient_;
    // Adadelta mu parameter
    double                  mu_{0.99};
    // prevent dividing with zero
    double                  epsilon_{1.e-09};
  };
  //
  //
  //
  template< typename Type >
  Alps::AdamGradient< Type, std::vector< Type >, std::vector< Type >, Alps::Arch::CPU >::AdamGradient()
  {
    //mini_batch_    = static_cast< std::size_t >(Alps::LoadDataSet::instance()->get_data()["mountain"]["strategy"]["mini_batch"]);
    learning_rate_ = static_cast< double >(Alps::LoadDataSet::instance()->get_data()["mountain"]["strategy"]["learning_rate"]);
  }
  //
  //
  //
  template< typename Type > void
  Alps::AdamGradient< Type,
				   std::vector< Type >,
				   std::vector< Type >,
				   Alps::Arch::CPU >::set_parameters( const std::size_t Current_size,
								      const std::size_t Prev_size )
  {
    delta_size_             = Current_size;
    delta_                  = std::vector< Type >( Current_size, 0. );
    cumul_squared_gradient_ = std::vector< Type >( Current_size, 0. );
  }
  //
  //
  //
  template< typename Type > void
  Alps::AdamGradient< Type, std::vector< Type >, std::vector< Type >, Alps::Arch::CPU >::reset_parameters()
  {
    delta_                  = std::vector< Type >( delta_size_, 0. );
    //cumul_squared_gradient_ = std::vector< Type >( delta_size_, 0. );
  }
  //
  //
  //
  template< typename Type > void
  Alps::AdamGradient< Type,
				   std::vector< Type >,
				   std::vector< Type >,
				   Alps::Arch::CPU >::add_tensors( const std::vector< Type >& Delta,
								   const std::vector< Type >& Z )
  {
    try
      {
	//
	//
	for ( std::size_t d = 0 ; d < delta_size_ ; d++ )
	  delta_[d] += Delta[d];
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
  Alps::AdamGradient< Type,
				   std::vector< Type >,
				   std::vector< Type >,
				   Alps::Arch::CPU >::solve( const bool Forced )
  {
    //
    // multiply all the elements with the learning rate
    double learn = learning_rate_;
    std::vector< Type > velocity = std::vector< Type >( delta_size_, 0. );
    //
    for ( int i = 0 ; i < delta_size_ ; ++i )
      {
	// Cumul the square of the gradient
	cumul_squared_gradient_[i] += delta_[i] * delta_[i];
	velocity[i] = - learning_rate_ * delta_[i] / sqrt( cumul_squared_gradient_[i] + epsilon_ );
      }

    //
    //
    return velocity;
  }
  /** \class AdamGradient
   *
   * \brief 
   * This class is the Adam class.
   * 
   * - MiniBatch: the size of the mini-bach
   *   * MiniBatch =  1 -- Adam
   *   * MiniBatch =  n -- batch of size n images, n < N the total 
   *                       number of images
   *   * MiniBatch = -1 -- the model uses all the images
   * 
   */
  template< typename Type >
  class AdamGradient< Type,
				   Eigen::MatrixXd,
				   Eigen::MatrixXd,
				   Alps::Arch::CPU > : public Alps::Gradient< Eigen::MatrixXd,
									      Eigen::MatrixXd >
  {
  public:
    /** Costructor */
    explicit AdamGradient();
    /** Destructor */
    virtual ~AdamGradient() = default;

      
    //
    // Accessors
    //
    // Set the layer sizes
    virtual void            set_parameters( const std::size_t, const std::size_t )        override;
    // Reset the layer sizes
    virtual void            reset_parameters()                                            override;
      
      
    //
    // Functions
    //
    // Get the type of optimizer
    virtual const Alps::Grad get_optimizer() const
    { return Alps::Grad::Adam;};
    // Add tensor elements
    virtual void            add_tensors( const Eigen::MatrixXd&, const Eigen::MatrixXd& ) override;
    // Backward propagation
    virtual Eigen::MatrixXd solve( const bool = false)                                    override;

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
    // number of weights
    std::size_t     delta_size_{0};
    // number of previous weights
    std::size_t     prev_size_{0};
    // delta_ = [W epsilon] * ...
    Eigen::MatrixXd delta_;
    // activation from the previous layer
    Eigen::MatrixXd previous_activation_;
  };
  //
  //
  //
  template< typename Type >
  Alps::AdamGradient< Type,
				   Eigen::MatrixXd,
				   Eigen::MatrixXd,
				   Alps::Arch::CPU >::AdamGradient()
  {
    //mini_batch_    = static_cast< std::size_t >(Alps::LoadDataSet::instance()->get_data()["mountain"]["strategy"]["mini_batch"]);
    learning_rate_ = static_cast< double >(Alps::LoadDataSet::instance()->get_data()["mountain"]["strategy"]["learning_rate"]);
  }
  //
  //
  //
  template< typename Type > void
  Alps::AdamGradient< Type,
				   Eigen::MatrixXd,
				   Eigen::MatrixXd,
				   Alps::Arch::CPU >::set_parameters( const std::size_t Current_size,
								      const std::size_t Prev_size )
  {
    delta_size_          = Current_size;
    prev_size_           = Prev_size;
    delta_               = Eigen::MatrixXd::Zero( Current_size, 1 );
    previous_activation_ = Eigen::MatrixXd::Zero( Prev_size, 1 );
  }
  //
  //
  //
  template< typename Type > void
  Alps::AdamGradient< Type,
				   Eigen::MatrixXd,
				   Eigen::MatrixXd,
				   Alps::Arch::CPU >::reset_parameters()
  {
    delta_               = Eigen::MatrixXd::Zero( delta_size_, 1 );
    previous_activation_ = Eigen::MatrixXd::Zero( prev_size_, 1 );
  }
  //
  //
  //
  template< typename Type > void
  Alps::AdamGradient< Type, Eigen::MatrixXd,
				   Eigen::MatrixXd,
				   Alps::Arch::CPU >::add_tensors( const Eigen::MatrixXd& Delta,
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
  Alps::AdamGradient< Type,
				   Eigen::MatrixXd,
				   Eigen::MatrixXd,
				   Alps::Arch::CPU >::solve( const bool Forced )
  {
	return - learning_rate_ * delta_ * previous_activation_.transpose();
  }
  /** \class AdamGradient
   *
   * \brief 
   * This class is the Adam class.
   * 
   * - MiniBatch: the size of the mini-bach
   *   * MiniBatch =  1 -- Adam
   *   * MiniBatch =  n -- batch of size n images, n < N the total 
   *                       number of images
   *   * MiniBatch = -1 -- the model uses all the images
   * 
   */
  template< typename Type, typename Tensor1_Type, typename Tensor2_Type >
  class AdamGradient< Type, Tensor1_Type, Tensor2_Type, Alps::Arch::CUDA > : public Alps::Gradient< Tensor1_Type, Tensor2_Type >
  {
  public:
    /** Costructor */
    explicit AdamGradient(){};
    /** Destructor */
    virtual ~AdamGradient() = default;

      
    //
    // Accessors
    //
    // Set the layer sizes
    virtual void            set_parameters( const std::size_t, const std::size_t )      override{};
    // Reset the layer sizes
    virtual void            reset_parameters()                                          override{};

      
    //
    // Functions
    //
    // Get the type of optimizer
    virtual const Alps::Grad get_optimizer() const
    { return Alps::Grad::Adam;};
    // Add tensor elements
    virtual void         add_tensors( const Tensor1_Type&, const Tensor1_Type& ) override {};
    // Backward propagation
    virtual Tensor2_Type solve( const bool = false )                                     override
    { return Tensor2_Type();};
  };
}
#endif
