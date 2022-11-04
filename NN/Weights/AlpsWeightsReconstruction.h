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
#ifndef ALPSWEIGHTSRECONSTRUCTION_H
#define ALPSWEIGHTSRECONSTRUCTION_H
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
  /** \class WeightsReconstruction
   *
   * \brief WeightsReconstruction represents the basic window element of the reconstruction layer.
   * 
   */
  template< typename Tensor1_Type,
	    Alps::Arch Architecture,
	    typename Activation,
	    typename Solver,
	    int Dim >
  class WeightsReconstruction : public Alps::Weights< Tensor1_Type, Tensor1_Type, Dim >
  {
    //
    // Aliases
    using LayerTensorsVec = std::vector< std::reference_wrapper< Alps::LayerTensors< Tensor1_Type, Dim > > >;
    using ActivationVec   = std::array < std::vector< Tensor1_Type >, 2 >;

    

    
  public:
    /** Constructor. */
    explicit WeightsReconstruction( const std::vector< int > ){};
    
    /** Destructor */
    virtual ~WeightsReconstruction() = default;


    //
    // Accessors
    //
    //! Activation tensor from the previous layer
    virtual void                                  set_activations( LayerTensorsVec&,
								   LayerTensorsVec& )         override;
    //! Get size of the tensor
    virtual const std::vector< std::size_t >      get_tensor_size() const noexcept            override
    { return std::vector< std::size_t >(); };						      
    //! Get the tensor			     						      
    virtual const std::vector< Tensor1_Type >&    get_tensor() const noexcept                 override
    { return weights_;};			     						      
    //! Update the tensor
    virtual std::vector< Tensor1_Type >&          update_tensor()                             override 
    { return weights_;};

    
    												      
    //												      
    // Functions										      
    //												      
    //! Save the weights										      
    virtual void                               save_tensor() const                            override{};
    //! Load the weights										      
    virtual void                               load_tensor( const std::string )               override{};
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
    std::vector< Tensor1_Type >     weights_;
    //! Weights activation.
    Activation                      activation_;
    //! The mountain observed: fully connected layer.
    const Alps::Layer&              layer_;
  };
  /** \class WeightsReconstruction
   *
   * \brief 
   * WeightsReconstruction object represents the basic window element of the reconstruction layer.
   * 
   */
  template< typename Type,
	    typename Activation,
	    typename Solver,
	    int Dim >
  class WeightsReconstruction< Type, Alps::Arch::CPU, Activation, Solver, Dim > : public Alps::Weights< Type, Type, Dim >
  {
    //
    // Aliases
    using LayerTensorsVec = std::vector< std::reference_wrapper< Alps::LayerTensors< Type, Dim > > >;
    using ActivationVec   = std::array < std::vector< Type >, 2 >;

    

    


  public:
    /** Constructor. */
    explicit WeightsReconstruction( const Alps::Layer& );
    
    /** Destructor */
    virtual ~WeightsReconstruction() = default;


    //
    // Accessors
    //
    
    /** @brief Set activation computes the gradient of the cost function.
     *
     * The gradient of the cost function is processed and will be used in the update of the weights.
     * 
     *  @param attached_layers: LayerTensorsVec& The reference to the current layer
     *  @param attached_layers: LayerTensorsVec& The reference to the previous layers
     *  @return void.
     */
    virtual void                                  set_activations( LayerTensorsVec&,
								   LayerTensorsVec& )         override;
    // Get size of the tensor
    virtual const std::vector< std::size_t >      get_tensor_size() const noexcept            override
    { return std::vector< std::size_t >(); };							      
    // Get the tensor										      
    virtual const std::vector< Type >&            get_tensor() const noexcept                 override
    { return weights_;};										      
    // Update the tensor
    virtual std::vector< Type >&                  update_tensor()                             override 
    { return weights_;};

    
    
    //												      
    // Functions										      
    //												      
    // Save the weights										      
    virtual void                                  save_tensor() const                         override{};
    // Load the weights										      
    virtual void                                  load_tensor( const std::string )            override{};
    
     /** @brief Activation calculation from the previous layers' attached to the current layer at the last layer.
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
    virtual ActivationVec                         activate( LayerTensorsVec& )                override;
    
    /** @brief Calculate the weight error.
     *
     * In the reconstruction layer, the weighted error is summarized by the error at the last layer 
     * (the current layer).
     *
     *  @param attached_layers: LayerTensorsVec& The reference to the current layers
     *  @param attached_layers: LayerTensorsVec& The reference to the previous layers
     *  @return void.
     */
    virtual void                                  weighted_error( LayerTensorsVec&,
								  LayerTensorsVec& )          override;
    // Update the weights
    virtual void                                  update()                                    override;
    // Forced the weight update
    virtual void                                  forced_update()                             override;





  private:
    // Reconstruction single weight
    std::vector< Type >                    weights_;
    // Output feature
    int                                    feature_{0};
    // weights activation
    Activation                             activation_;
    //
    // The mountain observed: fully connected layer
    const Alps::Layer&                     layer_;
    //
    // Type of gradient descent
    std::shared_ptr< Alps::Gradient_base > gradient_;
  };
  //
  //
  template< typename T, typename A, typename S, int D >
  WeightsReconstruction< T, Alps::Arch::CPU, A, S, D >::WeightsReconstruction( const Alps::Layer& Layer ):
    layer_{Layer}
  {
    try
      {
	//
	// Create a unique id for the layer
	std::random_device                   rd;
	std::mt19937                         generator( rd() );
	std::uniform_real_distribution< T >  distribution(-0.005,0.005);
	weights_.push_back( 0. /*distribution(generator)*/ );
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
	// There is only one weight for the reconstruction
	std::dynamic_pointer_cast< Alps::Gradient< std::vector< T >,
						   std::vector< T > > >(gradient_)->set_parameters( 1, 0 );
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
  template< typename T, typename A, typename S, int D > void
  WeightsReconstruction< T, Alps::Arch::CPU, A, S, D >::set_activations( LayerTensorsVec& Image_tensors,
									 LayerTensorsVec& Prev_image_tensors )
  {
    //
    //
    std::size_t layer_size = (Image_tensors[0].get()).get_image(TensorOrder1::ACTIVATION).get_tensor_size()[0];
    double de = 0.;
    //
    //
    for ( std::size_t d = 0 ; d < layer_size ; d++ )
      de += (Image_tensors[0].get())[TensorOrder1::ERROR][d];
    //
    std::vector< T > dE(1, de);
    //
    // process the weights
    // All the images from the batch (1 image, mini-batch or full batch) should have been processed
    std::dynamic_pointer_cast< Alps::Gradient< std::vector< T >,
					       std::vector< T > > >(gradient_)->reset_parameters();
    // We pass the gradient of the cost function
    std::dynamic_pointer_cast< Alps::Gradient< std::vector< T >,
					       std::vector< T > > >(gradient_)->add_tensors( dE,
											     std::vector<T>(1,1) );
  };
  //
  //
  //
  template< typename T, typename A, typename S, int D > std::array< std::vector< T >, 2 >
  WeightsReconstruction< T, Alps::Arch::CPU, A, S, D >::activate( LayerTensorsVec& Image_tensors )
  {
    //
    // features_number represents the number of layers attached to the current layer
    int
      features_number = Image_tensors.size(),
      size_in         = (Image_tensors[0].get()).get_tensor_size()[0];
    //
    std::vector< T > a_out(  size_in, 0. );
    std::vector< T > z_out(  size_in, 0. );
    std::vector< T > dz_out( size_in, 0. );

//    std::cout << "Reconstruction weights_[0]: "
//	      << weights_[0] << std::endl;
    //
    //
    try
      {
	//
	// compute the activation
	for ( int f = 0 ; f < features_number ; f++ )
	  {
	    //
	    // Check the size between the getting in layer and the number of colums are the same
	    std::size_t layer_size = (Image_tensors[f].get()).get_image(TensorOrder1::ACTIVATION).get_tensor_size()[0];
	    if ( layer_size != static_cast< std::size_t >(size_in) )
	      throw MAC::MACException( __FILE__, __LINE__,
				       "Error in the construction of the weight mastrix's dimensions.",
				       ITK_LOCATION );
	    //
	    //
	    for ( int s = 0 ; s < size_in ; s++ )
	      a_out[s] += (Image_tensors[f].get())[Alps::TensorOrder1::ACTIVATION][s];
	    
	  }
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit(-1);
      }
    //
    // Compute the feature activation
    for ( int s = 0 ; s < size_in ; s++ )
      a_out[s] +=  weights_[0];
    //
    std::vector< T > a_out_scaled = std::move( Alps::feature_scaling< T >(a_out, Alps::Scaling::NONE) );
    //
    for ( int s = 0 ; s < size_in ; s++ )
      {
	z_out[s]  = activation_.f(  a_out_scaled[s] );
	dz_out[s] = activation_.df( a_out_scaled[s] );
	//std::cout << "Recon: a_out["<<s<<"] = " << a_out[s] << " z_out["<<s<<"] = " << z_out[s] << std::endl;
      }

    //
    //
    return { z_out, dz_out };
  };
  //
  // The tensors size is the size of the weighted error tensor from the previous layer
  // The second input is the error tensor calculated at the present layer.
  template< typename T, typename A, typename S, int D > void
  WeightsReconstruction< T, Alps::Arch::CPU, A, S, D >::weighted_error( LayerTensorsVec& Prev_image_tensors,
									LayerTensorsVec& Image_tensors )
  {
    try
      {
	int
	  prev_features_number = Prev_image_tensors.size(),
	  size_in              = (Prev_image_tensors[0].get()).get_tensor_size()[0],
	  size_out             = (Image_tensors[0].get()).get_tensor_size()[0];
	//
	if ( size_in != size_out )
	  throw MAC::MACException( __FILE__, __LINE__,
				     "The input size must be the same as the output size.",
				     ITK_LOCATION );
	//
	for ( int k = 0 ; k < prev_features_number ; k++ )
	  for ( long int s = 0 ; s < size_in ; s++ )
	    (Prev_image_tensors[k].get())[TensorOrder1::WERROR][s] = (Image_tensors[0].get())[TensorOrder1::ERROR][s];
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
  template< typename T, typename A, typename S, int D > void
  WeightsReconstruction< T, Alps::Arch::CPU, A, S, D >::update()
  {
    weights_[0] += std::dynamic_pointer_cast< Alps::Gradient< std::vector< T >,
							      std::vector< T > > >(gradient_)->solve()[0];
  };
  //
  //
  //
  template< typename T, typename A, typename S, int D > void
  WeightsReconstruction< T, Alps::Arch::CPU, A, S, D >::forced_update()
  {
    weights_[0] += std::dynamic_pointer_cast< Alps::Gradient< std::vector< T >,
							      std::vector< T > > >(gradient_)->solve( true )[0];
  };
  /** \class WeightsReconstruction
   *
   * \brief 
   * WeightsReconstruction object represents the basic window element of the reconstruction layer.
   * 
   */
  template< typename Type1,
	    typename Activation,
	    typename Solver,
	    int Dim >
  class WeightsReconstruction< Type1, Alps::Arch::CUDA, Activation, Solver, Dim > : public Alps::Weights< Type1, Type1, Dim >
  {
    //
    // Aliases
    using LayerTensorsVec = std::vector< std::reference_wrapper< Alps::LayerTensors< Type1, Dim > > >;
    using ActivationVec   = std::array < std::vector< Type1 >, 2 >;






  public:
    /** Constructor. */
    explicit WeightsReconstruction( std::shared_ptr< Alps::Layer >  ){};
    
    /** Destructor */
    virtual ~WeightsReconstruction() = default;


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
    virtual const std::vector< Type1 >&           get_tensor() const noexcept                 override
    { return weights_;};										      
    //! Update the tensor
    virtual std::vector< Type1 >&                 update_tensor()                             override 
    { return weights_;};
												      
    												      
    //												      
    // Functions										      
    //												      
    //! Save the weights										      
    virtual void                                  save_tensor() const                         override{};
    //! Load the weights										      
    virtual void                                  load_tensor( const std::string )            override{};
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
    std::vector< Type1 >            weights_;
    //! weights activation
    Activation                      activation_;
    //
    //! The mountain observed: fully connected layer
    const Alps::Layer&              layer_;
  };
}
#endif
