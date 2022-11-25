/*=========================================================================
* Alps is a deep learning library approach customized for neuroimaging data 
* Copyright (C) 2021 Yann Cobigo (yann.cobigo@yahoo.com)
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <https://www.gnu.org/licenses/>.
*=========================================================================*/
#ifndef ALPSWEIGHTSCONVOLUTION_H
#define ALPSWEIGHTSCONVOLUTION_H
//
//
//
#include <iostream>
#include <bits/stdc++.h>
//
#include "MACException.h"
#include "AlpsTools.h"
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
   * \brief class WeightsConvolution represents the basic weight element 
   * of the convolution layer. 
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
    using LayerTensorsVec = std::vector< std::shared_ptr< Alps::LayerTensors< Tensor1_Type, Dim > > >;
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
    //! Activation tensor from the previous layer
    virtual void                               set_activations( LayerTensorsVec&,
								LayerTensorsVec& )            override{};
    //! Get size of the tensor
    virtual const std::vector< std::size_t >   get_tensor_size() const noexcept               override
    { return std::vector< std::size_t >(); };						      
    //! Get the tensor			     						      
    virtual const std::vector< Tensor2_Type >& get_tensor() const noexcept                    override
    { return weights_;};			     						      
    //! Update the tensor
    virtual std::vector< Tensor2_Type >&       update_tensor()                                override 
    { return weights_;};

    												      
    //												      
    // Functions										      
    //												      
    //! Save the weights										      
    virtual void                               save_tensor( std::ofstream& ) const            override{};
    //! Load the weights										      
    virtual void                               load_tensor( const std::ofstream& )            override{};
    //
    //
    //! Activate
    virtual ActivationVec                      activate( LayerTensorsVec& )                   override{};
    //! Weighted error
    virtual void                               weighted_error( LayerTensorsVec&,
							       LayerTensorsVec& )             override{};
    //! Update the weights
    virtual void                               update()                                       override{};
    //! Force the weight update
    virtual void                               forced_update()                                override{};




  private:
    //! Matrix of weigths.
    std::vector< Tensor2_Type >     weights_;
    //! Window for weigths.
    std::shared_ptr< Tensor2_Type > window_{nullptr};
    //! Weights activation.
    Activation                      activation_;
    //! The mountain observed: fully connected layer.
    const Alps::Layer&              layer_;
  };
  /** \class WeightsConvolution
   *
   * \brief WeightsConvolution represents the basic window element of the convolution layer.
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
    using LayerTensorsVec = std::vector< std::shared_ptr< Alps::LayerTensors< Type, Dim > > >;
    using ActivationVec   = std::array < std::vector< Type >, 2 >;

    



  public:
    /** Constructor. */
    explicit WeightsConvolution( const Alps::Layer&,
				 std::shared_ptr< Kernel >, const int );
    
    /** Destructor */
    virtual ~WeightsConvolution() = default;


    //
    // Accessors
    //
    //! Activation tensor from the previous layer
    virtual void                                   set_activations( LayerTensorsVec&,
								    LayerTensorsVec& )        override;
    // Get size of the tensor
    virtual const std::vector< std::size_t >       get_tensor_size() const noexcept           override
    { return std::vector< std::size_t >(); };							      
    //! Get the tensor										      
    virtual const std::vector< Kernel >&           get_tensor() const noexcept                override
    { return weights_;};										      
    //! Update the tensor
    virtual std::vector< Kernel >&                 update_tensor()                            override 
    { return weights_;};


    
    //												      
    // Functions										      
    //												      
    //! Save the weights										      
    virtual void                                   save_tensor( std::ofstream& ) const        override{};
    //! Load the weights										      
    virtual void                                   load_tensor( const std::ofstream& )        override{};

    /** @brief Activation calculation from the previous layers' attached to the current layer.
     *
     * The forward propagation is a matrix multiplication process of the weight tensor 
     * $W_{\nu n}^{\mu m}$ of the layer $\mu$, associated with the kernel $m$, with the flatten 
     * activation vector $z^{\nu n}$ of a connected layers $\nu$ to the layer $\mu$. 
     *
     * $$
     * a_{i}^{\mu m} = W_{\nu n}^{\mu m} z_{i}^{\nu n} + b^{\mu m}
     * $$
     *
     *  @param attached_layers: LayerTensorsVec& The reference to the previous layers
     *  @return ActivationVec. The two first elements: function activation <0> and 
     *                         derivative <1> are processed.
     */
    virtual ActivationVec                          activate( LayerTensorsVec& )               override;
     
    /** @brief Calculate the weight error.
     *
     * The weighted error from the next layer back propagated, and $f'$ has a dimension 
     * is the derivative of the activation function at the current layer:
     *
     * \varepsilon_{\mu m}^{(i)} = \left[ \varepsilon_{\delta d}^{(i)} W_{\mu m}^{\delta d} \right] \odot f'^{\mu m}
     *
     *  @param attached_layers: LayerTensorsVec& The reference to the previous layers
     *  @param attached_layers: LayerTensorsVec& The reference to the current layers
     *  @return void.
     */
    virtual void                                   weighted_error( LayerTensorsVec&,
								   LayerTensorsVec& )         override;
    //! Update the weights
    virtual void                                   update()                                   override;
    //! Forced the weight update
    virtual void                                   forced_update()                            override;





  private:
    // Matrix of weigths
    std::vector< Kernel >                  weights_;
    //! Window for weigths.
    std::shared_ptr< Kernel >              window_{nullptr};
    //! Output feature
    int                                    feature_{0};
    //! weights activation
    Activation                             activation_;
    //
    //! The mountain observed: fully connected layer
    const Alps::Layer&                     layer_;
    //
    // Type of gradient descent
    std::shared_ptr< Alps::Gradient_base > gradient_;
  };
  //
  //
  template< typename T, typename K, typename A, typename S, int D >
  WeightsConvolution< T, K, Alps::Arch::CPU, A, S, D >::WeightsConvolution( const Alps::Layer&   Layer,
									    std::shared_ptr< K > Window,
									    const int            Feature):
    window_{Window}, feature_{Feature}, layer_{Layer}
  {
    try
      {
	//
	// We use the non-transposed weights
	window_->set_transpose( false );

	//
	// Select the optimizer strategy
	S gradient;
	switch( gradient.get_optimizer() ) {
	case Alps::Grad::SGD:
	  {
	    gradient_ = std::make_shared< Alps::StochasticGradientDescent< double,
									   std::vector< T >,
									   std::vector< T >,
									   Alps::Arch::CPU > >();
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
	std::dynamic_pointer_cast< Alps::Gradient< std::vector< T >,
						   std::vector< T > > >(gradient_)->set_parameters( weight_number, 0 );
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
  WeightsConvolution< T, K, Alps::Arch::CPU, A, S, D >::set_activations( LayerTensorsVec& Prev_image_tensors,
									 LayerTensorsVec& Image_tensors )
  {
    //
    // retrieve the weight matrix
    const Eigen::SparseMatrix< int, Eigen::RowMajor >& matrix_weights   = window_->get_weights_matrix();
    const std::vector< std::vector< double > >&        deriv_weight_val = window_->get_derivated_weight_values();
    //
    int
      prev_features_number = Prev_image_tensors.size(),
      weight_number        = deriv_weight_val.size(),
      size_out             = matrix_weights.rows();

//    //
//    // Hadamard production between the weighted error and the
//    // derivative of the activation
//    std::vector< T > hadamard = std::move( (Image_tensors[feature_].get())( TensorOrder1::WERROR,
//									    TensorOrder1::DERIVATIVE) );
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
		for ( typename Eigen::SparseMatrix< int, Eigen::RowMajor >::InnerIterator it( matrix_weights, k);
		      it; ++it )
		  wz[k] += deriv_weight_val[w][ static_cast< int >(it.value()) ]
		    * ( *(Prev_image_tensors[f].get()) )[Alps::TensorOrder1::ACTIVATION][it.index()];
	    //
	    for ( int o = 0 ; o < size_out ; o++)
	      //de += hadamard[o] * wz[o];
	      de += ( *(Image_tensors[feature_].get()) )[TensorOrder1::ERROR][o] * wz[o];
	  }
	else
	  // Case for the bias
	  for ( int o = 0 ; o < size_out ; o++)
	    //de += hadamard[o];
	    de += ( *(Image_tensors[feature_].get()) )[TensorOrder1::ERROR][o];
	//
	//
	dE[w] = de; 
      }

    //
    // process
    std::dynamic_pointer_cast< Alps::Gradient< std::vector< T >,
					       std::vector< T > > >(gradient_)->add_tensors( dE,
											     std::vector<T>() );
  };
  //
  //
  //
  template< typename T, typename K, typename A, typename S, int D > std::array< std::vector< T >, 2 >
  WeightsConvolution< T, K, Alps::Arch::CPU, A, S, D >::activate( LayerTensorsVec& Image_tensors )
  {
//    //
//    // We use the non-transposed weights
//    window_->set_transpose( false );
    
    //
    // retrieve the weight matrix
    const Eigen::SparseMatrix< int, Eigen::RowMajor >& matrix_weights = window_->get_weights_matrix();
    const std::vector< double >&                       weight_val     = window_->get_convolution_weight_values( feature_ );

//    std::cout << matrix_weights << std::endl;
//    int w = 0;
//    for ( auto i : weight_val )
//      std::cout << "weight_values_[kernel: "
//		<<feature_<< "].get()[weight: "
//		<<w++<<"] = "
//		<< i
//		<< std::endl;

    
    //
    // features_number represents the number of layers attached to the current layer
    int
      features_number = Image_tensors.size(),
      size_in         = matrix_weights.cols(),
      size_out        = matrix_weights.rows();
    //
    std::vector< T > a_out(  size_out, 0. );
    std::vector< T > z_out(  size_out, 0. );
    std::vector< T > dz_out( size_out, 0. );
    //
    // compute the activation
    try
      {
	for ( int f = 0 ; f < features_number ; f++ )
	  {
	    //
	    // Check the size between the getting in layer and the number of colums are the same
	    std::size_t layer_size = ( *Image_tensors[f].get() ).get_image(TensorOrder1::ACTIVATION).get_tensor_size()[0];
	    if ( layer_size != static_cast< std::size_t >(size_in) )
	      throw MAC::MACException( __FILE__, __LINE__,
				       "Error in the construction of the weight mastrix's dimensions.",
				       ITK_LOCATION );
	    // 
	    for (int k = 0 ; k < matrix_weights.outerSize() ; ++k )
	      for ( typename Eigen::SparseMatrix< int, Eigen::RowMajor >::InnerIterator it( matrix_weights, k); it; ++it )
		a_out[k] += weight_val[static_cast< int >(it.value())]
		  * ( *(Image_tensors[f].get()) )[Alps::TensorOrder1::ACTIVATION][it.index()];
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
      a_out[s] +=  weight_val[0]; // add the bias
    //
    std::vector< T > a_out_scaled = std::move( Alps::feature_scaling< T >(a_out, Alps::Scaling::NONE) );
    //
    for ( int s = 0 ; s < size_out ; s++ )
      {
	z_out[s]  = activation_.f(  a_out_scaled[s] );  
	dz_out[s] = activation_.df( a_out_scaled[s] ); 
//	std::cout
//	  << "ConvolutionL a_out["<<s<<"] " << a_out_scaled[s]
//	  << " weight_val[0] " << weight_val[0]
//	  << " z_out["<<s<<"] " << z_out[s]
//	  << std::endl;
      }
    
    //
    //
    return { z_out, dz_out };
  };
  //
  //
  //
  template< typename T, typename K, typename A, typename S, int D > void
  WeightsConvolution< T, K, Alps::Arch::CPU, A, S, D >::weighted_error( LayerTensorsVec& Prev_image_tensors,
									LayerTensorsVec& Image_tensors )
  {
//    //
//    // We use the non-transposed weights
//    window_->set_transpose( false );

    //
    // retrieve the weight matrix
    const Eigen::SparseMatrix< int, Eigen::RowMajor >& matrix_weightsT = window_->get_weights_matrix().transpose();
    const std::vector< double >&                       weight_val      = window_->get_convolution_weight_values( feature_ );
    //
    int
      prev_features_number = Prev_image_tensors.size(),
      size_in              = matrix_weightsT.rows(),
      size_out             = matrix_weightsT.cols();

    //
    // Compute the error at the current layer using the weighted error computed at the next layer
    for ( int o = 0 ; o < size_out ; o++ )
      {
	( *(Image_tensors[feature_].get()) )[TensorOrder1::ERROR][o] = ( *(Image_tensors[feature_].get()) )[TensorOrder1::WERROR][o] * ( *(Image_tensors[feature_].get()) )[TensorOrder1::DERIVATIVE][o];
//	std::cout
//	  << "(Image_tensors[feature_].get())[TensorOrder1::WERROR]["<<o<<"] " << (Image_tensors[feature_].get())[TensorOrder1::WERROR][o]
//	  << "(Image_tensors[feature_].get())[TensorOrder1::DERIVATIVE][o] "   << (Image_tensors[feature_].get())[TensorOrder1::DERIVATIVE][o]
//	  << std::endl;
      }

    //
    // Compute the weighted error that will be used in the previous layer
    std::vector< T > we( size_in, 0. );
    //
    for (int k = 0 ; k < matrix_weightsT.outerSize() ; ++k )
      for ( typename Eigen::SparseMatrix< int, Eigen::RowMajor >::InnerIterator it( matrix_weightsT, k); it; ++it )
	we[k] += weight_val[ static_cast< int >(it.value()) ]
	  * ( *(Image_tensors[feature_].get()) )[TensorOrder1::ERROR][it.index()];
    // Replicate to all the previouse connected features' layers
    for ( int f = 0 ; f < prev_features_number ; ++f )
      for (int k = 0 ; k < size_in ; ++k )
    	( *(Prev_image_tensors[f].get()) )[TensorOrder1::WERROR][k] += we[k];
  };
  //
  //
  //
  template< typename T, typename K, typename A, typename S, int D > void
  WeightsConvolution< T, K, Alps::Arch::CPU, A, S, D >::update()
  {
    window_->set_convolution_weight_values( feature_,
					    std::dynamic_pointer_cast< Alps::Gradient< std::vector< T >,
					    std::vector< T > > >(gradient_)->solve() );
  };
  //
  //
  //
  template< typename T, typename K, typename A, typename S, int D > void
  WeightsConvolution< T, K, Alps::Arch::CPU, A, S, D >::forced_update()
  {
    window_->set_convolution_weight_values( feature_,
					    std::dynamic_pointer_cast< Alps::Gradient< std::vector< T >,
					    std::vector< T > > >(gradient_)->solve(true) );
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
    using LayerTensorsVec = std::vector< std::shared_ptr< Alps::LayerTensors< Type1, Dim > > >;
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
    //! Activation tensor from the previous layer
    virtual void                                  set_activation( LayerTensorsVec&,
								  LayerTensorsVec&)           override{};
    //! Get size of the tensor
    virtual const std::vector< std::size_t >      get_tensor_size() const noexcept            override
    { return std::vector< std::size_t >(); };							      
    //! Get the tensor										      
    virtual const std::vector< Type2 >&           get_tensor() const noexcept                 override
    { return weights_;};										      
    //! Update the tensor
    virtual std::vector< Type2 >&                 update_tensor()                             override 
    { return weights_;};


    
    												      
    //												      
    // Functions										      
    //												      
    //! Save the weights										      
    virtual void                                  save_tensor( std::ofstream& ) const         override{};
    //! Load the weights										      
    virtual void                                  load_tensor( const std::ofstream& )         override{};
    //
    //
    //! Activate
    virtual ActivationVec                         activate( LayerTensorsVec& )                override{};
    //! Weighted error
    virtual void                                  weighted_error( LayerTensorsVec&,
								  LayerTensorsVec& )          override{};
    //! Update the weights
    virtual void                                  update()                                    override{};
    //! Force the update of the weights
    virtual void                                  forced_update()                             override{};





  private:
    //! Matrix of weigths
    std::vector< Type2 >            weights_;
    //! Window for weigths.
    std::shared_ptr< Type2 >        window_{nullptr};
    //! weights activation
    Activation                      activation_;
    //
    //! The mountain observed: fully connected layer
    const Alps::Layer&              layer_;
  };
}
#endif
