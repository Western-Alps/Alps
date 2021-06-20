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
  class WeightsConvolution : public Alps::Weights< Tensor1_Type, Tensor2_Type, Dim >
  {
    //
    // Aliases
    using LayerTensorsVec = std::vector< Alps::LayerTensors< Tensor1_Type, Dim > >;
    using ActivationVec   = std::array < std::vector< Tensor1_Type >, 2 >;

    

    
  public:
    /** Constructor. */
    explicit WeightsConvolution( const std::vector< int >,
				 const std::vector< int >,
				 const std::vector< int > ){};
    
    /** Destructor */
    virtual ~WeightsConvolution() = default;


    //
    // Accessors
    //
    // Activation tensor from the previous layer
    virtual void                               set_activations( LayerTensorsVec&,
								LayerTensorsVec& )            override{};
    // Get size of the tensor
    virtual const std::vector< std::size_t >   get_tensor_size() const                        override
    { return std::vector< std::size_t >(); };						      
    // Get the tensor			     						      
    virtual const std::vector< Tensor2_Type >& get_tensor() const                             override
    { return weights_;};			     						      
    // Update the tensor
    virtual std::vector< Tensor2_Type >&       update_tensor()                                override 
    { return weights_;};
    //    // Set size of the tensor		     						      
    //    virtual void                               set_tensor_size( std::vector< std::size_t > )  override{};
    //    // Set the tensor			     						      
    //    virtual void                               set_tensor( std::shared_ptr< Tensor2_Type > )  override{};

    												      
    //												      
    // Functions										      
    //												      
    // Save the weights										      
    virtual void                               save_tensor() const                            override{};
    // Load the weights										      
    virtual void                               load_tensor( const std::string )               override{};
    //
    //
    // Activate
    virtual ActivationVec                      activate( LayerTensorsVec& )                   override{};
    // Weighted error
    virtual void                               weighted_error( LayerTensorsVec&,
							       LayerTensorsVec& )             override{};
    // Update the weights
    virtual void                               update()                                       override{};
    // Force the weight update
    virtual void                               forced_update()                                override{};




  private:
    //! Matrix of weigths.
    std::vector< Tensor2_Type >     weights_;
    //! Window for weigths.
    std::shared_ptr< Tensor2_Type > window_{nullptr};
    //! Weights activation.
    Activation                      activation_;
    //! The mountain observed: fully connected layer.
    std::shared_ptr< Alps::Layer >  layer_{nullptr};
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
  class WeightsConvolution< Type, Kernel, Alps::Arch::CPU, Activation, Solver, Dim > :
    public Alps::Weights< Type, Kernel, Dim >
  {
    //
    // Aliases
    using LayerTensorsVec = std::vector< Alps::LayerTensors< Type, Dim > >;
    using ActivationVec   = std::array < std::vector< Type >, 2 >;

    



  public:
    /** Constructor. */
    explicit WeightsConvolution( std::shared_ptr< Alps::Layer >,
				 std::shared_ptr< Kernel >, const int );
    
    /** Destructor */
    virtual ~WeightsConvolution() = default;


    //
    // Accessors
    //
    // Activation tensor from the previous layer
    virtual void                                   set_activations( LayerTensorsVec&,
								    LayerTensorsVec& )           override;
    // Get size of the tensor
    virtual const std::vector< std::size_t >       get_tensor_size() const                       override
    { return std::vector< std::size_t >(); };							      
    // Get the tensor										      
    virtual const std::vector< Kernel >&           get_tensor() const                            override
    { return weights_;};										      
    // Update the tensor
    virtual std::vector< Kernel >&                 update_tensor()                               override 
    { return weights_;};
    //    // Set size of the tensor									      
    //    virtual void                                   set_tensor_size( std::vector< std::size_t > ) override{};
    //    // Set the tensor										      
    //    virtual void                                   set_tensor( std::shared_ptr< Kernel > )       override{};

    
    //												      
    // Functions										      
    //												      
    // Save the weights										      
    virtual void                                   save_tensor() const                           override{};
    // Load the weights										      
    virtual void                                   load_tensor( const std::string )              override{};
    //
    //
    // Activate
    virtual ActivationVec                          activate( LayerTensorsVec& )                  override;
    // Weighted error
    virtual void                                   weighted_error( LayerTensorsVec&,
								   LayerTensorsVec& )            override;
    // Update the weights
    virtual void                                   update()                                      override;
    // Forced the weight update
    virtual void                                   forced_update()                               override;





  private:
    // Matrix of weigths
    std::vector< Kernel >                  weights_;
    //! Window for weigths.
    std::shared_ptr< Kernel >              window_{nullptr};
    // Output feature
    int                                    feature_{0};
    // weights activation
    Activation                             activation_;
    //
    // The mountain observed: fully connected layer
    std::shared_ptr< Alps::Layer >         layer_;
    //
    // Type of gradient descent
    //std::shared_ptr< Alps::Gradient_base > gradient_;
    Alps::StochasticGradientDescent< double,
									   std::vector< Type >,
									   std::vector< Type >,
									   Alps::Arch::CPU > gradient_;
  };
  //
  //
  template< typename T, typename K, typename A, typename S, int D >
  WeightsConvolution< T, K, Alps::Arch::CPU, A, S, D >::WeightsConvolution( std::shared_ptr< Alps::Layer > Layer,
									    std::shared_ptr< K >           Window,
									    const int                      Feature):
    window_{Window}, feature_{Feature}, layer_{Layer}
  {
    try
      {
	//
	// Select the optimizer strategy
	S gradient;
	switch( gradient.get_optimizer() ) {
	case Alps::Grad::SGD:
	  {
//	    gradient_ = std::make_shared< Alps::StochasticGradientDescent< double,
//									   std::vector< T >,
//									   std::vector< T >,
//									   Alps::Arch::CPU > >();
	    //
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
	std::size_t weight_number = window_->get_derivated_weight_values().size();
//	std::dynamic_pointer_cast< Alps::Gradient< std::vector< T >,
//						   std::vector< T > > >(gradient_)->set_parameters( weight_number, 0 );
	
	(gradient_).set_parameters( weight_number, 0 );
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
  WeightsConvolution< T, K, Alps::Arch::CPU, A, S, D >::set_activations( std::vector< Alps::LayerTensors< T, D > >& Prev_image_tensors,
									 std::vector< Alps::LayerTensors< T, D > >& Image_tensors )
  {
    //
    // We use the non-transposed weights
    window_->set_transpose( false );

    //
    // retrieve the weight matrix
    Eigen::SparseMatrix< int, Eigen::RowMajor > matrix_weights   = window_->get_weights_matrix();
    std::vector< double >                       weight_val       = window_->get_convolution_weight_values( feature_ );
    std::vector< std::vector< double > >        deriv_weight_val = window_->get_derivated_weight_values();
    //
    int
      prev_features_number = Prev_image_tensors.size(),
      weight_number        = deriv_weight_val.size(),
      size_in              = matrix_weights.cols(),
      size_out             = matrix_weights.rows();

    //
    // Hadamard production between the weighted error and the
    // derivative of the activation
    std::vector< T > hadamard = std::move( Image_tensors[feature_](TensorOrder1::WERROR, TensorOrder1::DERIVATIVE) );

    
    std::cout
      << "WeightsTransposedConvolution::set_activations" 
      << " layer_size[ACTIVATION]: " << Prev_image_tensors[0].get_image(TensorOrder1::ACTIVATION).get_tensor_size()[0]
      << " layer_size[ERROR]: " << Prev_image_tensors[0].get_image(TensorOrder1::ERROR).get_tensor_size()[0]
      << " layer_size[WERROR]: " << Prev_image_tensors[0].get_image(TensorOrder1::WERROR).get_tensor_size()[0]
      << " layer_size[ACTIVATION]: " << Image_tensors[0].get_image(TensorOrder1::ACTIVATION).get_tensor_size()[0]
      << " layer_size[ERROR]: " << Image_tensors[0].get_image(TensorOrder1::ERROR).get_tensor_size()[0]
      << " layer_size[WERROR]: " << Image_tensors[0].get_image(TensorOrder1::WERROR).get_tensor_size()[0]
      << " size_in: " << size_in
      << " size_out: " << size_out
      << std::endl;
    //
    // Replicate to all the previouse connected features' layers
    std::vector< T > dE( weight_number, 0. );
    //
    for ( int w = 0 ; w < weight_number ; w++ )
      {
	// update of the gradient
	double de = 0;
	//
	if ( w > 0 )
	  {
	    //
	    std::vector< T > wz( size_out, 0. );
	    for ( int f = 0 ; f < prev_features_number ; f++ )
	      for (int k = 0 ; k < matrix_weights.outerSize() ; ++k )
		for ( typename Eigen::SparseMatrix< int, Eigen::RowMajor >::InnerIterator it( matrix_weights, k); it; ++it )
		  wz[k] += deriv_weight_val[w][ static_cast< int >(it.value()) ]
		    * Prev_image_tensors[f][Alps::TensorOrder1::ACTIVATION][it.index()];
	    //
	    for ( int o = 0 ; o < size_out ; o++)
	      de += hadamard[o] * wz[o];
	  }
	else
	  // Case for the bias
	  for ( int o = 0 ; o < size_out ; o++)
	    de += hadamard[o];
	//
	//
	dE[w] = de; 
      }

    //
    // process
//    std::dynamic_pointer_cast< Alps::Gradient< std::vector< T >,
//					       std::vector< T > > >(gradient_)->add_tensors( dE, std::vector<T>() );
    (gradient_).add_tensors( dE, std::vector<T>() );
  };
  //
  //
  //
  template< typename T, typename K, typename A, typename S, int D > std::array< std::vector< T >, 2 >
  WeightsConvolution< T, K, Alps::Arch::CPU, A, S, D >::activate( std::vector< Alps::LayerTensors< T, D > >& Image_tensors )
  {
    //
    // We use the non-transposed weights
    window_->set_transpose( false );
    
    //
    // retrieve the weight matrix
    Eigen::SparseMatrix< int, Eigen::RowMajor > matrix_weights = window_->get_weights_matrix();
    std::vector< double >                       weight_val     = window_->get_convolution_weight_values( feature_ );
    //
    int
      features_number = Image_tensors.size(),
      size_in         = matrix_weights.cols(),
      size_out        = matrix_weights.rows();
    //
    std::vector< T > a_out( size_out, 0. );
    std::vector< T > z_out( size_out, 0. );
    std::vector< T > dz_out( size_out, 0. );
    std::cout
      << "WeightsConvolution::activate" 
      << " layer_size[ACTIVATION]: " << Image_tensors[0].get_image(TensorOrder1::ACTIVATION).get_tensor_size()[0]
      << " layer_size[ERROR]: " << Image_tensors[0].get_image(TensorOrder1::ERROR).get_tensor_size()[0]
      << " layer_size[WERROR]: " << Image_tensors[0].get_image(TensorOrder1::WERROR).get_tensor_size()[0]
      << " size_in: " << size_in
      << " size_out: " << size_out
      << std::endl;
    //
    // compute the activation
    try
      {
	for ( int f = 0 ; f < features_number ; f++ )
	  {
	    //
	    // Check the size between the getting in layer and the number of colums are the same
	    std::size_t layer_size = Image_tensors[f].get_image(TensorOrder1::ACTIVATION).get_tensor_size()[0];
	    if ( layer_size != static_cast< std::size_t >(size_in) )
	      throw MAC::MACException( __FILE__, __LINE__,
				       "Error in the construction of the weight mastrix's dimensions.",
				       ITK_LOCATION );
	    // 
	    for (int k = 0 ; k < matrix_weights.outerSize() ; ++k )
	      for ( typename Eigen::SparseMatrix< int, Eigen::RowMajor >::InnerIterator it( matrix_weights, k); it; ++it )
		a_out[k] += weight_val[static_cast< int >(it.value())]
		  * Image_tensors[f][Alps::TensorOrder1::ACTIVATION][it.index()];
	  }

      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit(-1);
      }
    //
    // Compute the feature activation
    for ( int s = 0 ; s < size_out ; s++ )
      {
	z_out[s]  = activation_.f( a_out[s] + weight_val[0] );  // add the bias
	dz_out[s] = activation_.df( a_out[s] + weight_val[0] ); // add the bias
      }
    
    //
    //
    return { z_out, dz_out };
  };
  //
  //
  //
  template< typename T, typename K, typename A, typename S, int D > void
  WeightsConvolution< T, K, Alps::Arch::CPU, A, S, D >::weighted_error( std::vector< Alps::LayerTensors< T, D > >& Prev_image_tensors,
									std::vector< Alps::LayerTensors< T, D > >& Image_tensors )
  {
    //
    // We use the non-transposed weights
    window_->set_transpose( false );

    //
    // retrieve the weight matrix
    Eigen::SparseMatrix< int, Eigen::RowMajor > matrix_weights = window_->get_weights_matrix().transpose();
    std::vector< double >                       weight_val     = window_->get_convolution_weight_values( feature_ );
    //
    int
      prev_features_number = Prev_image_tensors.size(),
      size_in              = matrix_weights.rows(),
      size_out             = matrix_weights.cols();
    //
    std::vector< T > we( size_in, 0. );
    std::cout
      << "WeightsConvolution::weighted_error" 
      << " layer_size[ACTIVATION]: " << Prev_image_tensors[0].get_image(TensorOrder1::ACTIVATION).get_tensor_size()[0]
      << " layer_size[ERROR]: " << Prev_image_tensors[0].get_image(TensorOrder1::ERROR).get_tensor_size()[0]
      << " layer_size[WERROR]: " << Prev_image_tensors[0].get_image(TensorOrder1::WERROR).get_tensor_size()[0]
      << " size_in: " << size_in
      << " size_out: " << size_out
      << std::endl;
    //
    // compute the activation
    for (int k = 0 ; k < matrix_weights.outerSize() ; ++k )
      for ( typename Eigen::SparseMatrix< int, Eigen::RowMajor >::InnerIterator it( matrix_weights, k); it; ++it )
	we[k] += weight_val[ static_cast< int >(it.value()) ]
	  * Image_tensors[feature_][TensorOrder1::ERROR][it.index()];
    // Replicate to all the previouse connected features' layers
    for ( int f = 0 ; f < prev_features_number ; ++f )
      for (int k = 0 ; k < size_in ; ++k )
    	Prev_image_tensors[f][TensorOrder1::WERROR][k] += we[k];
  };
  //
  //
  //
  template< typename T, typename K, typename A, typename S, int D > void
  WeightsConvolution< T, K, Alps::Arch::CPU, A, S, D >::update()
  {
//    window_->set_convolution_weight_values( feature_,
//					    std::dynamic_pointer_cast< Alps::Gradient< std::vector< T >,
//					    std::vector< T > > >(gradient_)->solve() );
    window_->set_convolution_weight_values( feature_,
					    (gradient_).solve() );
  };
  //
  //
  //
  template< typename T, typename K, typename A, typename S, int D > void
  WeightsConvolution< T, K, Alps::Arch::CPU, A, S, D >::forced_update()
  {
//    window_->set_convolution_weight_values( feature_,
//					    std::dynamic_pointer_cast< Alps::Gradient< std::vector< T >,
//					    std::vector< T > > >(gradient_)->solve(true) );
    window_->set_convolution_weight_values( feature_,
					    (gradient_).solve(true) );
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
  class WeightsConvolution< Type1, Type2, Alps::Arch::CUDA, Activation, Solver, Dim > : public Alps::Weights< Type1, Type2, Dim >
  {
    //
    // Aliases
    using LayerTensorsVec = std::vector< Alps::LayerTensors< Type1, Dim > >;
    using ActivationVec   = std::array < std::vector< Type1 >, 2 >;






  public:
    /** Constructor. */
    explicit WeightsConvolution( std::shared_ptr< Alps::Layer >,
				 const std::vector< int >,
				 const std::vector< int >,
				 const std::vector< int > ){};
    
    /** Destructor */
    virtual ~WeightsConvolution() = default;


    //
    // Accessors
    //
    // Activation tensor from the previous layer
    virtual void                                  set_activation( LayerTensorsVec&,
								  LayerTensorsVec&)               override{};
    // Get size of the tensor
    virtual const std::vector< std::size_t >      get_tensor_size() const                        override
    { return std::vector< std::size_t >(); };							      
    // Get the tensor										      
    virtual const std::vector< Type2 >&           get_tensor() const                             override
    { return weights_;};										      
    // Update the tensor
    virtual std::vector< Type2 >&                 update_tensor()                                override 
    { return weights_;};
    //    // Set size of the tensor									      
    //    virtual void                                  set_tensor_size( std::vector< std::size_t > )  override{};
    //    // Set the tensor										      
    //    virtual void                                  set_tensor( std::shared_ptr< Type2 > )         override{};
												      
    												      
    //												      
    // Functions										      
    //												      
    // Save the weights										      
    virtual void                                  save_tensor() const                            override{};
    // Load the weights										      
    virtual void                                  load_tensor( const std::string )               override{};
    //
    //
    // Activate
    virtual ActivationVec                         activate( LayerTensorsVec& )                   override{};
    // Weighted error
    virtual void                                  weighted_error( LayerTensorsVec&,
								  LayerTensorsVec& )             override{};
    // Update the weights
    virtual void                                  update()                                       override{};
    // Force the update of the weights
    virtual void                                  forced_update()                                override{};





  private:
    // Matrix of weigths
    std::vector< Type2 >            weights_;
    //! Window for weigths.
    std::shared_ptr< Type2 >        window_{nullptr};
    // weights activation
    Activation                      activation_;
    //
    // The mountain observed: fully connected layer
    std::shared_ptr< Alps::Layer >  layer_;
  };
}
#endif
