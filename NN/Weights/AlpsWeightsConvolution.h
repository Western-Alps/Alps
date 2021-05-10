#ifndef ALPSWEIGHTSCONVOLUTION_H
#define ALPSWEIGHTSCONVOLUTION_H
//
//
//
#include <iostream>
#include <bits/stdc++.h>
//
#include "MACException.h"
#include "AlpsWeights.h"
#include "AlpsLayer.h"
#include "AlpsSGD.h"
//
//
//
namespace Alps
{
  /** \class WeightsConvolution
   *
   * \brief 
   * WeightsConvolution object represents the basic window element of the convolution layer.
   * 
   */
  template< typename Tensor1_Type,
	    typename Tensor2_Type,
	    Alps::Arch Architecture,
	    typename Activation,
	    typename Solver,
	    int Dim >
  class WeightsConvolution : public Alps::Weights< Tensor1_Type, Tensor2_Type >
  {
    //
    // 
  public:
    /** Constructor. */
    explicit WeightsConvolution( const std::vector< int >,
				 const std::vector< int >,
				 const std::vector< int > ){};
    
    /** Destructor */
    virtual ~WeightsConvolution(){};


    //
    // Accessors
    //
    // Activation tensor from the previous layer
    virtual void set_activations( std::vector< Alps::LayerTensors< Tensor1_Type, Dim > >&,
				  std::vector< Alps::LayerTensors< Tensor1_Type, Dim > >& ) override{};
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
			 std::shared_ptr< Tensor1_Type > > activate( std::vector< Alps::LayerTensors< Tensor1_Type, Dim > >& )       override{};
    // Weighted error
    virtual void                                           weighted_error( std::vector< Alps::LayerTensors< Tensor1_Type, Dim > >&,
									   std::vector< Alps::LayerTensors< Tensor1_Type, Dim > >& ) override{};
    // Update the weights
    virtual void                               update()                                               override{};
    // Force the weight update
    virtual void                               forced_update()                                        override{};




  private:
    //! Matrix of weigths.
    std::shared_ptr< Tensor2_Type >   weights_{nullptr};
    //! Weights activation.
    Activation                        activation_;
    //! The mountain observed: fully connected layer.
    std::shared_ptr< Alps::Layer >    layer_{nullptr};
  };
  /** \class WeightsConvolution
   *
   * \brief 
   * WeightsConvolution object represents the basic window element of the convolution layer.
   * 
   */
  template< typename Type,
	    typename Kernel,
	    typename Activation,
	    typename Solver,
	    int Dim >
  class WeightsConvolution< Type, Kernel, Alps::Arch::CPU, Activation, Solver, Dim > : public Alps::Weights< Type, Kernel >
  {
    //
    // 
  public:
    /** Constructor. */
    explicit WeightsConvolution( std::shared_ptr< Alps::Layer >,
				 std::shared_ptr< Kernel >, const int );
    
    /** Destructor */
    virtual ~WeightsConvolution(){};


    //
    // Accessors
    //
    // Activation tensor from the previous layer
    virtual void set_activations( std::vector< Alps::LayerTensors< Type, Dim > >&,
				  std::vector< Alps::LayerTensors< Type, Dim > >& )                    override;
    // Get size of the tensor
    virtual const std::vector< std::size_t >       get_tensor_size() const                             override
    { return std::vector< std::size_t >(); };							      
    // Get the tensor										      
    virtual std::shared_ptr< Kernel >              get_tensor() const                                  override
    {return weights_;};										      
    // Set size of the tensor									      
    virtual void                                   set_tensor_size( std::vector< std::size_t > )       override{};
    // Set the tensor										      
    virtual void                                   set_tensor( std::shared_ptr< Kernel > )             override{};

    
    //												      
    // Functions										      
    //												      
    // Save the weights										      
    virtual void                                   save_tensor() const                                 override{};
    // Load the weights										      
    virtual void                                   load_tensor( const std::string )                    override{};
    //
    //
    // Activate
    virtual std::tuple < std::shared_ptr< Type >,
			 std::shared_ptr< Type > > activate( std::vector< Alps::LayerTensors< Type, Dim > >& )       override;
    // Weighted error
    virtual void                                   weighted_error( std::vector< Alps::LayerTensors< Type, Dim > >&,
								   std::vector< Alps::LayerTensors< Type, Dim > >& ) override;
    // Update the weights
    virtual void                                   update()                                            override;
    // Forced the weight update
    virtual void                                   forced_update()                                     override;





  private:
    // Matrix of weigths
    std::shared_ptr< Kernel >              weights_;
    // Output feature
    int                                    feature_{0};
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
  template< typename T, typename K, typename A, typename S, int D >
  WeightsConvolution< T, K, Alps::Arch::CPU, A, S, D >::WeightsConvolution( std::shared_ptr< Alps::Layer > Layer,
									    std::shared_ptr< K >           Window,
									    const int                      Feature):
    weights_{Window}, feature_{Feature}, layer_{Layer}
  {
    try
      {
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
//	//
//	//
//	std::dynamic_pointer_cast< Alps::Gradient< Eigen::MatrixXd,
//						   Eigen::MatrixXd > >(gradient_)->set_parameters( current_nodes,
//												   previous_nodes + 1 );
//	
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
  template< typename T, typename K, typename A, typename S, int D > void
  WeightsConvolution< T, K, Alps::Arch::CPU, A, S, D >::set_activations( std::vector< Alps::LayerTensors< T, D > >& Image_tensors,
									 std::vector< Alps::LayerTensors< T, D > >& Prev_image_tensors )
  {
    long int
      prev_tensors_size = 0,
      tensors_size      = 0;
    //
    // retrieve the weight matrix
    Eigen::SparseMatrix< int, Eigen::RowMajor > matrix_weights = weights_->get_weights_matrix();
    std::shared_ptr< double >                   weight_val     = weights_->get_convolution_weight_values( feature_ );
    //
    int
      features_number = Image_tensors.size(),
      size_in         = matrix_weights.cols(),
      size_out        = matrix_weights.rows();
    //
    //
    try
      {
	//
	// The weights belong to the layer
	std::string layer_name = layer_->get_layer_name();
	
	//
	// Check the dimensions are right
	for ( auto tensor : Prev_image_tensors )
	  prev_tensors_size += static_cast< long int >( tensor.get_tensor_size()[0] );
	//
	for ( auto tensor : Image_tensors )
	  tensors_size += static_cast< long int >( tensor.get_tensor_size()[0] );
	//
	if ( size_out != tensors_size && size_in != prev_tensors_size + 1 )
	  {
	    std::string
	      mess = std::string("There is mismatch between the weight dimensions [")
	      + std::to_string( size_out ) + std::string(",") + std::to_string(size_in  ) 
	      + std::string("] and the layer dimensions: [")
	      + std::to_string( tensors_size )
	      + std::string(",") + std::to_string( prev_tensors_size )
	      + std::string("+1.]"); 
	    throw MAC::MACException( __FILE__, __LINE__,
				     mess.c_str(),
				     ITK_LOCATION );
	  }
//
//	
//	//
//	// Create the activation and error to update the weights
//	Eigen::MatrixXd z     = Eigen::MatrixXd::Zero( prev_tensors_size + 1, 1 );
//	Eigen::MatrixXd delta = Eigen::MatrixXd::Zero( tensors_size, 1 );
//	//
//	// Load the tensor previouse image into a Eigen vector
//	std::size_t shift = 1;
//	z(0,0) = 1.; // bias
//	for ( auto tensor : Prev_image_tensors )
//	  {
//	    std::size_t img_size = tensor.get_tensor_size()[0];
//	       for ( std::size_t s = 0 ; s < img_size ; s++ )
//		 z(s+shift,0) = tensor[TensorOrder1::ACTIVATION][s];
//	       shift += img_size;
//	  }
//	// treatment is sligtly different whether we are on the last layer or not
//	if ( layer_name == "__output_layer__" )
//	  {
//	   // Load the error tensor of the current image into a Eigen vector
//	   for ( auto tensor : Image_tensors )
//	     {
//	      std::size_t img_size = tensor.get_tensor_size()[0];
//	      for ( std::size_t s = 0 ; s < img_size ; s++ )
//		delta(s,0) = tensor[TensorOrder1::ERROR][s];
//	      shift += img_size;
//	     }
//	  }
//	else
//	  {
//	    // Hadamart production between the weighted error and the
//	    // derivative of the activation
//	    std::shared_ptr< T > hadamart = (Image_tensors[0])( TensorOrder1::WERROR, TensorOrder1::DERIVATIVE );
//	    // Load the error tensor of the current image into a Eigen vector
//	    for ( auto tensor : Image_tensors )
//	      {
//		std::size_t img_size = tensor.get_tensor_size()[0];
//		for ( std::size_t s = 0 ; s < img_size ; s++ )
//		  delta(s,0) = hadamart.get()[s];
//		shift += img_size;
//	      }
//	  }
//	// process
//	std::dynamic_pointer_cast< Alps::Gradient< Eigen::MatrixXd,
//						   Eigen::MatrixXd > >(gradient_)->add_tensors( delta, z );
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
  template< typename T, typename K, typename A, typename S, int D > std::tuple< std::shared_ptr< T >,
										std::shared_ptr< T > >
  WeightsConvolution< T, K, Alps::Arch::CPU, A, S, D >::activate( std::vector< Alps::LayerTensors< T, D > >& Image_tensors )
  {
    try
      {
	// ToDo: add some checks
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit(-1);
      }
    //
    // retrieve the weight matrix
    Eigen::SparseMatrix< int, Eigen::RowMajor > matrix_weights = weights_->get_weights_matrix();
    std::shared_ptr< double >                   weight_val     = weights_->get_convolution_weight_values( feature_ );
    //
    int
      features_number = Image_tensors.size(),
      size_out        = matrix_weights.rows();
    //
    std::shared_ptr< T > a_out  = std::shared_ptr< T >( new  T[size_out](), //-> init to 0
							std::default_delete< T[] >() );
    std::shared_ptr< T > z_out  = std::shared_ptr< T >( new  T[size_out](), //-> init to 0
							std::default_delete< T[] >() );
    std::shared_ptr< T > dz_out = std::shared_ptr< T >( new  T[size_out](), //-> init to 0
							std::default_delete< T[] >() );
    // compute the activation
    for ( int f = 0 ; f < features_number ; f++ )
      for (int k = 0 ; k < matrix_weights.outerSize() ; ++k )
	for ( typename Eigen::SparseMatrix< int, Eigen::RowMajor >::InnerIterator it( matrix_weights, k); it; ++it )
	  a_out.get()[k] += weight_val.get()[ static_cast< int >(it.value()) ]
	    * Image_tensors[f][Alps::TensorOrder1::ACTIVATION][it.index()];
    //
    // Compute the feature activation
    for ( int s = 0 ; s < size_out ; s++ )
      {
	z_out.get()[s]  = activation_.f( a_out.get()[s] + weight_val.get()[0] );  // add the bias
	dz_out.get()[s] = activation_.df( a_out.get()[s] + weight_val.get()[0] ); // add the bias
      }
    
    //
    //
    return std::make_tuple( z_out, dz_out );
  };
  //
  //
  //
  template< typename T, typename K, typename A, typename S, int D > void
  WeightsConvolution< T, K, Alps::Arch::CPU, A, S, D >::weighted_error( std::vector< Alps::LayerTensors< T, D > >& Prev_image_tensors,
									std::vector< Alps::LayerTensors< T, D > >& Image_tensors )
  {
    long int
      prev_tensors_size = 0,
      tensors_size      = 0;
    //
    // retrieve the weight matrix
    Eigen::SparseMatrix< int, Eigen::RowMajor > matrix_weights = weights_->get_weights_matrix();
    std::shared_ptr< double >                   weight_val     = weights_->get_convolution_weight_values( feature_ );
    //
    int
      features_number = Image_tensors.size(),
      size_in         = matrix_weights.cols(),
      size_out        = matrix_weights.rows();
    //
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
	if ( size_out != tensors_size && size_in != prev_tensors_size + 1 )
	  {
	    std::string
	      mess = std::string("There is mismatch between the weight dimensions [")
	      + std::to_string( size_out ) + std::string(",") + std::to_string(size_in  ) 
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
    // compute the activation
    // ToDo: make sure we use all the kernels and the product appear on a transpose vector with the lines
    // of the weight matrix
    for ( int f = 0 ; f < features_number ; f++ )
      for (int k = 0 ; k < matrix_weights.outerSize() ; ++k )
	for ( typename Eigen::SparseMatrix< int, Eigen::RowMajor >::InnerIterator it( matrix_weights, k); it; ++it )
	  Prev_image_tensors[0][TensorOrder1::WERROR][k] = weight_val.get()[ static_cast< int >(it.value()) ]
	    * Image_tensors[0][TensorOrder1::ERROR][k];
  };
  //
  //
  //
  template< typename T, typename K, typename A, typename S, int D > void
  WeightsConvolution< T, K, Alps::Arch::CPU, A, S, D >::update()
  {
//    *(weights_.get()) += std::dynamic_pointer_cast< Alps::Gradient< Eigen::MatrixXd,
//								    Eigen::MatrixXd > >(gradient_)->solve();
  };
  //
  //
  //
  template< typename T, typename K, typename A, typename S, int D > void
  WeightsConvolution< T, K, Alps::Arch::CPU, A, S, D >::forced_update()
  {
//    *(weights_.get()) += std::dynamic_pointer_cast< Alps::Gradient< Eigen::MatrixXd,
//								    Eigen::MatrixXd > >(gradient_)->solve( true );
  };
  /** \class WeightsConvolution
   *
   * \brief 
   * WeightsConvolution object represents the basic window element of the convolution layer.
   * 
   */
  template< typename Type1,
	    typename Type2,
	    typename Activation,
	    typename Solver,
	    int Dim >
  class WeightsConvolution< Type1, Type2, Alps::Arch::CUDA, Activation, Solver, Dim > : public Alps::Weights< Type1, Type2 >
  {
    //
    // 
  public:
    /** Constructor. */
    explicit WeightsConvolution( std::shared_ptr< Alps::Layer >,
				 const std::vector< int >,
				 const std::vector< int >,
				 const std::vector< int > ){};
    
    /** Destructor */
    virtual ~WeightsConvolution(){};


    //
    // Accessors
    //
    // Activation tensor from the previous layer
    virtual void set_activation( std::vector< Alps::LayerTensors< Type1, Dim > >&,
				 std::vector< Alps::LayerTensors< Type1, Dim > >&) override{};
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
			 std::shared_ptr< Type1 > > activate( std::vector< Alps::LayerTensors< Type1, Dim > >& )       override{};
    // Weighted error
    virtual void                                    weighted_error( std::vector< Alps::LayerTensors< Type1, Dim > >&,
								    std::vector< Alps::LayerTensors< Type1, Dim > >& ) override{};
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
