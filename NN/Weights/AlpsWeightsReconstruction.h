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
   * \brief 
   * WeightsReconstruction object represents the basic window element of the reconstruction layer.
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
    using LayerTensorsVec = std::vector< Alps::LayerTensors< Tensor1_Type, Dim > >;
    using ActivationVec   = std::array < std::vector< Tensor1_Type >, 2 >;

    

    
  public:
    /** Constructor. */
    explicit WeightsReconstruction( const std::vector< int > ){};
    
    /** Destructor */
    virtual ~WeightsReconstruction(){};


    //
    // Accessors
    //
    // Activation tensor from the previous layer
    virtual void                                  set_activations( LayerTensorsVec&,
								LayerTensorsVec& )            override;
    // Get size of the tensor
    virtual const std::vector< std::size_t >      get_tensor_size() const                     override
    { return std::vector< std::size_t >(); };						      
    // Get the tensor			     						      
    virtual const std::vector< Tensor1_Type >&    get_tensor() const                          override
    { return weights_;};			     						      
    // Update the tensor
    virtual std::vector< Tensor1_Type >&          update_tensor()                             override 
    { return weights_;};
//    // Set size of the tensor		     						      
//    virtual void                               set_tensor_size( std::vector< std::size_t > )  override{};
//    // Set the tensor			     						      
//    virtual void                               set_tensor( std::shared_ptr< Tensor1_Type > )  override{};

    												      
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
    std::vector< Tensor1_Type >     weights_;
    //! Weights activation.
    Activation                      activation_;
    //! The mountain observed: fully connected layer.
    std::shared_ptr< Alps::Layer >  layer_{nullptr};
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
    using LayerTensorsVec = std::vector< Alps::LayerTensors< Type, Dim > >;
    using ActivationVec   = std::array < std::vector< Type >, 2 >;

    

    


  public:
    /** Constructor. */
    explicit WeightsReconstruction( std::shared_ptr< Alps::Layer > );
    
    /** Destructor */
    virtual ~WeightsReconstruction(){};


    //
    // Accessors
    //
    // Activation tensor from the previous layer
    virtual void                                  set_activations( LayerTensorsVec&,
								   LayerTensorsVec& )            override;
    // Get size of the tensor
    virtual const std::vector< std::size_t >      get_tensor_size() const                        override
    { return std::vector< std::size_t >(); };							      
    // Get the tensor										      
    virtual const std::vector< Type >&            get_tensor() const                             override
    { return weights_;};										      
    // Update the tensor
    virtual std::vector< Type >&                  update_tensor()                                override 
    { return weights_;};
//    // Set size of the tensor									      
//    virtual void                                  set_tensor_size( std::vector< std::size_t > )  override{};
//    // Set the tensor										      
//    virtual void                                  set_tensor( std::shared_ptr< Type > )          override{};

    
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
    virtual ActivationVec                         activate( LayerTensorsVec& )                   override;
    // Weighted error
    virtual void                                  weighted_error( LayerTensorsVec&,
								  LayerTensorsVec& )             override;
    // Update the weights
    virtual void                                  update()                                       override;
    // Forced the weight update
    virtual void                                  forced_update()                                override;





  private:
    // Reconstruction single weight
    std::vector< Type >                    weights_;
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
  template< typename T, typename A, typename S, int D >
  WeightsReconstruction< T, Alps::Arch::CPU, A, S, D >::WeightsReconstruction( std::shared_ptr< Alps::Layer > Layer ):
    layer_{Layer}
  {
    try
      {
	//
	// Create a unique id for the layer
	std::random_device                   rd;
	std::mt19937                         generator( rd() );
	std::normal_distribution< T >        distribution(0.0,1.0);
	weights_.push_back(  distribution(generator) );
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
//	std::dynamic_pointer_cast< Alps::Gradient< std::vector< T >,
//						   std::vector< T > > >(gradient_)->set_parameters( 1, 0 );
	(gradient_).set_parameters( 1, 0 );
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
  WeightsReconstruction< T, Alps::Arch::CPU, A, S, D >::set_activations( std::vector< Alps::LayerTensors< T, D > >& Image_tensors,
									 std::vector< Alps::LayerTensors< T, D > >& Prev_image_tensors )
  {
    //
    //
    std::size_t layer_size = Image_tensors[0].get_image(TensorOrder1::ACTIVATION).get_tensor_size()[0];
    double de = 0.;
    //
    for ( int d = 0 ; d < layer_size ; d++ )
      de += Image_tensors[0].get_image(TensorOrder1::ERROR).get_tensor_size()[0];
    //
    std::vector< T > dE(1, de);
    //
    //process
//    std::dynamic_pointer_cast< Alps::Gradient< std::vector< T >,
//					       std::vector< T > > >(gradient_)->add_tensors( dE, std::vector<T>() );
    (gradient_).add_tensors( dE, std::vector<T>() );
  };
  //
  //
  //
  template< typename T, typename A, typename S, int D > std::array< std::vector< T >, 2 >
  WeightsReconstruction< T, Alps::Arch::CPU, A, S, D >::activate( std::vector< Alps::LayerTensors< T, D > >& Image_tensors )
  {
    //
    //
    int
      features_number = Image_tensors.size(),
      size_in         = Image_tensors[0].get_tensor_size()[0];
    //
    std::vector< T > a_out( size_in, 0. );
    std::vector< T > z_out( size_in, 0. );
    std::vector< T > dz_out( size_in, 0. );
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
	    std::size_t layer_size = Image_tensors[f].get_image(TensorOrder1::ACTIVATION).get_tensor_size()[0];
	    if ( layer_size != size_in )
	      throw MAC::MACException( __FILE__, __LINE__,
				       "Error in the construction of the weight mastrix's dimensions.",
				       ITK_LOCATION );
	    //
	    //
	    for ( int s = 0 ; s < size_in ; s++ )
	      a_out[s] += Image_tensors[f][Alps::TensorOrder1::ACTIVATION][s];
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
      {
	z_out[s]  = activation_.f(  a_out[s] + weights_[0] );
	dz_out[s] = activation_.df( a_out[s] + weights_[0] );
      }

    //
    //
    return { z_out, dz_out };
  };
  //
  // The tensors size is the size of the weighted error tensor from the previous layer
  // The second input is the error tensor calculated at the present layer.
  template< typename T, typename A, typename S, int D > void
  WeightsReconstruction< T, Alps::Arch::CPU, A, S, D >::weighted_error( std::vector< Alps::LayerTensors< T, D > >& Prev_image_tensors,
									std::vector< Alps::LayerTensors< T, D > >& Image_tensors )
  {
//    int
//      prev_features_number = Prev_image_tensors.size(),
//      size_in              = Image_tensors[0].get_tensor_size()[0];
//    //
//    for ( int k = 0 ; k < prev_features_number ; k++ )
//      for ( long int s = 0 ; s < size_in ; s++ )
//	Prev_image_tensors[k][TensorOrder1::WERROR][s] = Image_tensors[0][TensorOrder1::ERROR][s];
  };
  //
  //
  //
  template< typename T, typename A, typename S, int D > void
  WeightsReconstruction< T, Alps::Arch::CPU, A, S, D >::update()
  {
//    weights_[0] += std::dynamic_pointer_cast< Alps::Gradient< std::vector< T >,
//							      std::vector< T > > >(gradient_)->solve()[0];
    weights_[0] += (gradient_).solve()[0];
  };
  //
  //
  //
  template< typename T, typename A, typename S, int D > void
  WeightsReconstruction< T, Alps::Arch::CPU, A, S, D >::forced_update()
  {
//    weights_[0] += std::dynamic_pointer_cast< Alps::Gradient< std::vector< T >,
//							      std::vector< T > > >(gradient_)->solve( true )[0];
    weights_[0] += (gradient_).solve( true )[0];
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
    using LayerTensorsVec = std::vector< Alps::LayerTensors< Type1, Dim > >;
    using ActivationVec   = std::array < std::vector< Type1 >, 2 >;






  public:
    /** Constructor. */
    explicit WeightsReconstruction( std::shared_ptr< Alps::Layer >  ){};
    
    /** Destructor */
    virtual ~WeightsReconstruction(){};


    //
    // Accessors
    //
    // Activation tensor from the previous layer
    virtual void                                  set_activation( LayerTensorsVec&,
								  LayerTensorsVec&)              override{};
    // Get size of the tensor
    virtual const std::vector< std::size_t >      get_tensor_size() const                        override
    { return std::vector< std::size_t >(); };							      
    // Get the tensor										      
    virtual const std::vector< Type1 >&           get_tensor() const                             override
    { return weights_;};										      
    // Update the tensor
    virtual std::vector< Type1 >&                 update_tensor()                                override 
    { return weights_;};
//    // Set size of the tensor									      
//    virtual void                                  set_tensor_size( std::vector< std::size_t > )  override{};
//    // Set the tensor										      
//    virtual void                                  set_tensor( std::shared_ptr< Type1 > )         override{};
												      
    												      
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
    std::vector< Type1 >            weights_;
    // weights activation
    Activation                      activation_;
    //
    // The mountain observed: fully connected layer
    std::shared_ptr< Alps::Layer >  layer_;
  };
}
#endif
