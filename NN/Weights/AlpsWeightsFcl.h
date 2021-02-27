#ifndef ALPSWEIGHTSFCL_H
#define ALPSWEIGHTSFCL_H
//
//
//
#include <iostream> 
#include <vector>
#include <memory>
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
//
#include "MACException.h"
#include "AlpsWeights.h"
#include "AlpsLayer.h"
#include "AlpsLayer.h"
#include "AlpsSGD.h"
//#include "AlpsClimber.h"
//#include "AlpsMountain.h"
//
//
//
/*! \namespace Alps
 *
 * Name space for Alps.
 *
 */
namespace Alps
{
  /*! \class WeightsFullyConnected
   * \brief class represents the weights container for fully
   * connected layers (FCL).
   *
   */
  template< typename Tensor1_Type,
	    typename Tensor2_Type,
	    Alps::Arch Architecture,
	    typename Activation,
	    typename Solver >
  class WeightsFcl : public Alps::Weights< Tensor1_Type, Tensor2_Type >
  {
  public:
    /** Constructor. */
    explicit WeightsFcl( std::shared_ptr< Alps::Layer >,
			 const std::vector< std::size_t >,
			 const std::vector< std::size_t > ){};
    /** Destructor */
    virtual ~WeightsFcl(){};


    //
    // Accessors
    //
    // Activation tensor from the previous layer
    virtual void set_activations( std::vector< Alps::LayerTensors< Tensor1_Type, 2 > >&,
				  std::vector< Alps::LayerTensors< Tensor1_Type, 2 > >& ) override{};
    // Get size of the tensor
    virtual const std::vector< std::size_t >   get_tensor_size() const                                override
    { return std::vector< std::size_t >(); };						      
    // Get the tensor			     						      
    virtual std::shared_ptr< Tensor2_Type >    get_tensor() const                                     override
    { return weights_;};			     						      
    // Set size of the tensor		     						      
    virtual void                               set_tensor_size( std::vector< std::size_t > )          override{};
    // Set the tensor			     						      
    virtual void                               set_tensor( std::shared_ptr< Tensor2_Type > )          override{};

    												      
    //												      
    // Functions										      
    //												      
    // Save the weights										      
    virtual void                               save_tensor() const                                    override{};
    // Load the weights										      
    virtual void                               load_tensor( const std::string )                       override{};
    //
    //
    // Activate
    virtual std::tuple < std::shared_ptr< Tensor1_Type >,
			 std::shared_ptr< Tensor1_Type > > activate( std::vector< Alps::LayerTensors< Tensor1_Type, 2 > >& )       override{};
    // Weighted error
    virtual void                                           weighted_error( std::vector< Alps::LayerTensors< Tensor1_Type, 2 > >&,
									   std::vector< Alps::LayerTensors< Tensor1_Type, 2 > >& ) override{};
    // Update the weights
    virtual void                               update()                                               override{};
    // Force the weight update
    virtual void                               forced_update()                                               override{};



  private:
    //! Matrix of weigths
    std::shared_ptr< Tensor2_Type >   weights_{nullptr};
    //! Weights activation
    Activation                        activation_;
    //! 
    
    // The mountain observed: fully connected layer
    std::shared_ptr< Alps::Layer >    layer_{nullptr};
  };
  /*! \class WeightsFullyConnected
   * \brief class representing the weights container for fully
   * connected layers (FCL) using CPU.
   *
   */
  template< typename Type,
	    typename Activation,
	    typename Solver >
  class WeightsFcl< Type, Eigen::MatrixXd, Alps::Arch::CPU, Activation, Solver > : public Alps::Weights< Type, Eigen::MatrixXd >
  {
  public:
    /** Constructor. */
    explicit WeightsFcl( std::shared_ptr< Alps::Layer >,
			 const std::vector< std::size_t >,
			 const std::vector< std::size_t > );
    /** Destructor */
    virtual ~WeightsFcl(){};


    //
    // Accessors
    //
    // Activation tensor from the previous layer
    virtual void set_activations( std::vector< Alps::LayerTensors< Type, 2 > >&,
				  std::vector< Alps::LayerTensors< Type, 2 > >& )               override;
    // Get size of the tensor
    virtual const std::vector< std::size_t >      get_tensor_size() const                             override
    { return std::vector< std::size_t >(); };							      
    // Get the tensor										      
    virtual std::shared_ptr< Eigen::MatrixXd >    get_tensor() const                                  override
    {return weights_;};										      
    // Set size of the tensor									      
    virtual void                                  set_tensor_size( std::vector< std::size_t > )       override{};
    // Set the tensor										      
    virtual void                                  set_tensor( std::shared_ptr< Eigen::MatrixXd > )    override{};

    
    //												      
    // Functions										      
    //												      
    // Save the weights										      
    virtual void                                  save_tensor() const                                 override{};
    // Load the weights										      
    virtual void                                  load_tensor( const std::string )                    override{};
    //
    //
    // Activate
    virtual std::tuple < std::shared_ptr< Type >,
			 std::shared_ptr< Type > > activate( std::vector< Alps::LayerTensors< Type, 2 > >& )       override;
    // Weighted error
    virtual void                                   weighted_error( std::vector< Alps::LayerTensors< Type, 2 > >&,
								   std::vector< Alps::LayerTensors< Type, 2 > >& ) override;
    // Update the weights
    virtual void                                   update()                                            override;
    // Forced the weight update
    virtual void                                   forced_update()                                            override;



  private:
    // Matrix of weigths
    std::shared_ptr< Eigen::MatrixXd >     weights_;
    // weights activation
    Activation                             activation_;
    //
    // The mountain observed: fully connected layer
    std::shared_ptr< Alps::Layer >         layer_;
    //
    // Type of gradient descent
    std::shared_ptr< Alps::Gradient_base > gradient_;
  };
  //
  //
  //
  template< typename T, typename A, typename S >
  Alps::WeightsFcl< T, Eigen::MatrixXd, Alps::Arch::CPU, A, S >::WeightsFcl( std::shared_ptr< Alps::Layer >    Layer,
									     const std::vector< std::size_t >  Layer_size,
									     const std::vector< std::size_t >  Prev_layer_size  ):
    layer_{Layer}
  {
    try
      {
	//
	// Get the number of layers defined
	std::size_t
	  prev_num_of_layers    = Prev_layer_size.size(),
	  current_num_of_layers = Layer_size.size();
	//
	//
	int
	  current_nodes  = 0,
	  previous_nodes = 0;
	// How many nodes we had in this layer:
	for ( std::size_t l = 0 ; l < current_num_of_layers ; l++ )
	  current_nodes += Layer_size[l];
	// How many nodes we had in the previous layer:
	for ( std::size_t pl = 0 ; pl < prev_num_of_layers ; pl++ )
	  previous_nodes += Prev_layer_size[pl];
	// Random create the variables between [-1,1]
	weights_ = std::make_shared< Eigen::MatrixXd >(Eigen::MatrixXd::Random( current_nodes,
										previous_nodes + 1 /* biais */) );
	std::cout
	  << "Weights ["<<weights_->rows()<<"x"<<weights_->cols()<<"]" 
	  << std::endl;

	//
	// Select the optimizer strategy
	S gradient;
	switch( gradient.get_optimizer() ) {
	case Alps::Grad::SGD:
	  {
	    gradient_ = std::make_shared< Alps::StochasticGradientDescent< double,
									   Eigen::MatrixXd,
									   Eigen::MatrixXd,
									   Alps::Arch::CPU > >();
	    
	    break;
	  };
	case Alps::Grad::MOMENTUM:
	case Alps::Grad::ADAGRAD:
	case Alps::Grad::Adam:
	case Alps::Grad::UNKNOWN:
	default:
	  {
	    std::string
	      mess = std::string("The optimizer has not been implemented yet.");
	    throw MAC::MACException( __FILE__, __LINE__,
				     mess.c_str(),
				     ITK_LOCATION );
	  }
	}
	//
	//
	std::dynamic_pointer_cast< Alps::Gradient< Eigen::MatrixXd,
						   Eigen::MatrixXd > >(gradient_)->set_parameters( current_nodes,
												   previous_nodes + 1 );

      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
      }
  };
  //
  //
  //
  template< typename T, typename A, typename S > void
  Alps::WeightsFcl< T, Eigen::MatrixXd, Alps::Arch::CPU, A, S >::set_activations( std::vector< Alps::LayerTensors< T, 2 > >& Image_tensors,
										  std::vector< Alps::LayerTensors< T, 2 > >& Prev_image_tensors )
  {
    try
      {
	//
	// The weights belong to the layer
	std::string layer_name = layer_->get_layer_name();
	
	//
	// Check the dimensions are right
	long int
	  tensors_size      = 0,
	  prev_tensors_size = 0;
	for ( auto tensor : Image_tensors )
	  tensors_size += static_cast< long int >( tensor.get_tensor_size()[0] );
	for ( auto tensor : Prev_image_tensors )
	  prev_tensors_size += static_cast< long int >( tensor.get_tensor_size()[0] );
	//
	if ( weights_->rows() != tensors_size || weights_->cols() != prev_tensors_size + 1  /*bias*/ )
	  {
	    std::string
	      mess = std::string("There is miss match between the weight matrix [(")
	      + std::to_string( weights_->rows() ) + std::string(",") + std::to_string( weights_->cols() )
	      + std::string(") and the size of the tensors (") + std::to_string( tensors_size )
	      + std::string(",") + std::to_string( prev_tensors_size ) + std::string("+1).");
	    throw MAC::MACException( __FILE__, __LINE__,
				     mess.c_str(),
				     ITK_LOCATION );
	  }

	
	//
	// Create the activation and error to update the weights
	Eigen::MatrixXd z     = Eigen::MatrixXd::Zero( prev_tensors_size + 1, 1 );
	Eigen::MatrixXd delta = Eigen::MatrixXd::Zero( tensors_size, 1 );
	//
	// Load the tensor previouse image into a Eigen vector
	std::size_t shift = 1;
	z(0,0) = 1.; // bias
	for ( auto tensor : Prev_image_tensors )
	  {
	    std::size_t img_size = tensor.get_tensor_size()[0];
	       for ( std::size_t s = 0 ; s < img_size ; s++ )
		 z(s+shift,0) = tensor[TensorOrder1::ACTIVATION][s];
	       shift += img_size;
	  }
	// treatment is sligtly different whether we are on the last layer or not
	if ( layer_name == "__output_layer__" )
	  {
	   // Load the error tensor of the current image into a Eigen vector
	   for ( auto tensor : Image_tensors )
	     {
	      std::size_t img_size = tensor.get_tensor_size()[0];
	      for ( std::size_t s = 0 ; s < img_size ; s++ )
		delta(s,0) = tensor[TensorOrder1::ERROR][s];
	      shift += img_size;
	     }
	  }
	else
	  {
	    // Hadamart production between the weighted error and the
	    // derivative of the activation
	    std::shared_ptr< T > hadamart = (Image_tensors[0])( TensorOrder1::WERROR, TensorOrder1::DERIVATIVE );
	    // Load the error tensor of the current image into a Eigen vector
	    for ( auto tensor : Image_tensors )
	      {
		std::size_t img_size = tensor.get_tensor_size()[0];
		for ( std::size_t s = 0 ; s < img_size ; s++ )
		  delta(s,0) = hadamart.get()[s];
		shift += img_size;
	      }
	  }
	// process
	std::dynamic_pointer_cast< Alps::Gradient< Eigen::MatrixXd,
						   Eigen::MatrixXd > >(gradient_)->add_tensors( delta, z );
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit(-1);
      }
  };
  //
  //
  //
  template< typename T, typename A, typename S > std::tuple< std::shared_ptr< T >,
							     std::shared_ptr< T > >
  Alps::WeightsFcl< T, Eigen::MatrixXd, Alps::Arch::CPU, A, S >::activate( std::vector< Alps::LayerTensors< T, 2 > >& Image_tensors )
  {
    try
      {
	//
	// Check the dimensions are right
	long int tensors_size = 0;
	for ( auto tensor : Image_tensors )
	  tensors_size += static_cast< long int >( tensor.get_tensor_size()[0] );
	//
	if ( weights_->cols() != tensors_size + 1 /*bias*/ )
	  {
	    std::string
	      mess = std::string("There is miss match between the number of columns (")
	      + std::to_string( weights_->cols() )
	      + std::string(") and the size of the input tensor (")
	      + std::to_string( tensors_size )
	      + std::string("+1).");
	    throw MAC::MACException( __FILE__, __LINE__,
				     mess.c_str(),
				     ITK_LOCATION );
	  }
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit(-1);
      }

    //
    // reset the z_in and save the information in the object
    Eigen::MatrixXd      z_in   = Eigen::MatrixXd::Zero( weights_->cols(), 1 );
    // Converter the tensor into an Eigen matrix
    std::shared_ptr< T > z_out  = std::shared_ptr< T >( new  T[weights_->rows()],
							std::default_delete< T[] >() );
    std::shared_ptr< T > dz_out = std::shared_ptr< T >( new  T[weights_->rows()],
							std::default_delete< T[] >() );
    Eigen::MatrixXd      a_out  = Eigen::MatrixXd::Zero( weights_->rows(), 1 );
    // Load the tensor image into a Eigen vector
    std::size_t shift = 1;
    z_in(0,0) = 1.; // bias
    for ( auto tensor : Image_tensors )
      {
	std::size_t img_size = tensor.get_tensor_size()[0];
	for ( std::size_t s = 0 ; s < img_size ; s++ )
	  z_in(s+shift,0) = tensor[TensorOrder1::ACTIVATION][s];
	shift += img_size;
      }
    // process
    a_out = *( weights_.get() ) * z_in;
    // Apply the activation function
    long int activation_size = weights_->rows();
    for ( long int s = 0 ; s < activation_size ; s++ )
      {
	z_out.get()[s]  = activation_.f( a_out(s,0) );
	dz_out.get()[s] = activation_.df( a_out(s,0) );
      }

    //
    //
    return std::make_tuple( z_out, dz_out );
  };
  //
  // The tensors size is the size of the weighted error tensor from the previous layer
  // The second input is the error tensor calculated at the present layer.
  template< typename T, typename A, typename S > void
  Alps::WeightsFcl< T, Eigen::MatrixXd, Alps::Arch::CPU, A, S >::weighted_error( std::vector< Alps::LayerTensors< T, 2 > >& Prev_image_tensors,
										 std::vector< Alps::LayerTensors< T, 2 > >& Image_tensors )
  {
    long int
      prev_tensors_size = 0,
      tensors_size      = 0;
    try
      {
	//
	// Check the dimensions are right
	for ( auto tensor : Prev_image_tensors )
	  prev_tensors_size += static_cast< long int >( tensor.get_tensor_size()[0] );
	//
	for ( auto tensor : Image_tensors )
	  tensors_size += static_cast< long int >( tensor.get_tensor_size()[0] );
	//
	if ( weights_->rows() != tensors_size && weights_->cols() != prev_tensors_size + 1 )
	  {
	    std::string
	      mess = std::string("There is miss match between the weight dimensions ")
	      + std::to_string( weights_->rows() ) + std::string(",") + std::to_string( weights_->cols() ) 
	      + std::string("] and the layer dimensions: [")
	      + std::to_string( tensors_size )
	      + std::string(",") + std::to_string( prev_tensors_size )
	      + std::string("+1.]"); 
	    throw MAC::MACException( __FILE__, __LINE__,
				     mess.c_str(),
				     ITK_LOCATION );
	  }
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit(-1);
      }

    
    //
    // reset the z_in and save the information in the object
    Eigen::MatrixXd error_in = Eigen::MatrixXd::Zero( tensors_size, 1 );
    Eigen::MatrixXd we_out   = Eigen::MatrixXd::Zero( prev_tensors_size + 1, 1 );
    // Load the image error tensor into a Eigen vector
    for ( long int s = 0 ; s < tensors_size ; s++ )
      error_in(s,0) = Image_tensors[0][TensorOrder1::ERROR][s];
    //
    // process
    we_out = error_in.transpose() * ( *(weights_.get()) );
    // We skip the bias
    // We add new weighted error for all the layers attached to
    // the previous layer
    for ( long int s = 0 ; s < prev_tensors_size ; s++ )
      Prev_image_tensors[0][TensorOrder1::WERROR][s] += we_out(s+1,0);
  };
  //
  //
  //
  template< typename T, typename A, typename S > void
  Alps::WeightsFcl< T, Eigen::MatrixXd, Alps::Arch::CPU, A, S >::update()
  {
    *(weights_.get()) += std::dynamic_pointer_cast< Alps::Gradient< Eigen::MatrixXd,
								    Eigen::MatrixXd > >(gradient_)->solve();
  };
  //
  //
  //
  template< typename T, typename A, typename S > void
  Alps::WeightsFcl< T, Eigen::MatrixXd, Alps::Arch::CPU, A, S >::forced_update()
  {
    *(weights_.get()) += std::dynamic_pointer_cast< Alps::Gradient< Eigen::MatrixXd,
								    Eigen::MatrixXd > >(gradient_)->solve( true );
  };
  /*! \class WeightsFullyConnected
   * \brief class representing the weights container for fully
   * connected layers (FCL) using CUDA.
   *
   */
  template< typename Type1,
	    typename Type2,
	    typename Activation,
	    typename Solver >
  class WeightsFcl< Type1, Type2, Alps::Arch::CUDA, Activation, Solver > : public Alps::Weights< Type1, Type2 >
  {
  public:
    /** Constructor. */
    explicit WeightsFcl( std::shared_ptr< Alps::Layer >,
			 const std::vector< std::size_t >,
			 const std::vector< std::size_t > ){std::cout << "CUDA treatment" << std::endl;};
    /** Destructor */
    virtual ~WeightsFcl(){};


    //
    // Accessors
    //
    // Activation tensor from the previous layer
    virtual void set_activation( std::vector< Alps::LayerTensors< Type1, 2 > >&,
				 std::vector< Alps::LayerTensors< Type1, 2 > >&) override{};
    // Get size of the tensor
    virtual const std::vector< std::size_t >      get_tensor_size() const                              override
    { return std::vector< std::size_t >(); };							      
    // Get the tensor										      
    virtual std::shared_ptr< Type2 >              get_tensor() const                                   override
    {return weights_;};										      
    // Set size of the tensor									      
    virtual void                                  set_tensor_size( std::vector< std::size_t > )        override{};
    // Set the tensor										      
    virtual void                                  set_tensor( std::shared_ptr< Type2 > )               override{};
												      
    												      
    //												      
    // Functions										      
    //												      
    // Save the weights										      
    virtual void                            save_tensor() const                                        override{};
    // Load the weights										      
    virtual void                            load_tensor( const std::string )                           override{};
    //
    //
    // Activate
    virtual std::tuple < std::shared_ptr< Type1 >,
			 std::shared_ptr< Type1 > > activate( std::vector< Alps::LayerTensors< Type1, 2 > >& )       override{};
    // Weighted error
    virtual void                                    weighted_error( std::vector< Alps::LayerTensors< Type1, 2 > >&,
								    std::vector< Alps::LayerTensors< Type1, 2 > >& ) override{};
    // Update the weights
    virtual void                            update()                                                   override{};
    // Force the update of the weights
    virtual void                            forced_update()                                                   override{};



  private:
    // Matrix of weigths
    std::shared_ptr< Type2 >           weights_;
    // weights activation
    Activation                         activation_;
    //
    // The mountain observed: fully connected layer
    std::shared_ptr< Alps::Layer >     layer_;
  };
}
#endif
