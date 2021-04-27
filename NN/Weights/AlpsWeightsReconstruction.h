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
  class WeightsReconstruction : public Alps::Weights< Tensor1_Type, Tensor1_Type >
  {
    //
    // 
  public:
    /** Constructor. */
    explicit WeightsReconstruction( const std::vector< int > ){};
    
    /** Destructor */
    virtual ~WeightsReconstruction(){};


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
    virtual std::shared_ptr< Tensor1_Type >    get_tensor() const                                     override
    { return weights_;};			     						      
    // Set size of the tensor		     						      
    virtual void                               set_tensor_size( std::vector< std::size_t > )          override{};
    // Set the tensor			     						      
    virtual void                               set_tensor( std::shared_ptr< Tensor1_Type > )          override{};

    												      
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
    std::shared_ptr< Tensor1_Type >   weights_{nullptr};
    //! Weights activation.
    Activation                        activation_;
    //! The mountain observed: fully connected layer.
    std::shared_ptr< Alps::Layer >    layer_{nullptr};
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
  class WeightsReconstruction< Type, Alps::Arch::CPU, Activation, Solver, Dim > : public Alps::Weights< Type, Type >
  {
    //
    // 
  public:
    /** Constructor. */
    explicit WeightsReconstruction( std::shared_ptr< Alps::Layer > );
    
    /** Destructor */
    virtual ~WeightsReconstruction(){};


    //
    // Accessors
    //
    // Activation tensor from the previous layer
    virtual void set_activations( std::vector< Alps::LayerTensors< Type, Dim > >&,
				  std::vector< Alps::LayerTensors< Type, Dim > >& )                    override{};
    // Get size of the tensor
    virtual const std::vector< std::size_t >       get_tensor_size() const                             override
    { return std::vector< std::size_t >(); };							      
    // Get the tensor										      
    virtual std::shared_ptr< Type >                get_tensor() const                                  override
    {return weights_;};										      
    // Set size of the tensor									      
    virtual void                                   set_tensor_size( std::vector< std::size_t > )       override{};
    // Set the tensor										      
    virtual void                                   set_tensor( std::shared_ptr< Type > )               override{};

    
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
								   std::vector< Alps::LayerTensors< Type, Dim > >& ) override{};
    // Update the weights
    virtual void                                   update()                                            override{};
    // Forced the weight update
    virtual void                                   forced_update()                                     override{};





  private:
    // Reconstruction single weight
    std::shared_ptr< Type >                weights_;
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
	weights_ = std::make_shared< T >( distribution(generator) );
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
  template< typename T, typename A, typename S, int D > std::tuple< std::shared_ptr< T >,
								    std::shared_ptr< T > >
  WeightsReconstruction< T, Alps::Arch::CPU, A, S, D >::activate( std::vector< Alps::LayerTensors< T, D > >& Image_tensors )
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
	int
	  features_number = Image_tensors.size(),
	  size_out        = Image_tensors[0].get_tensor_size()[0];
	//
	std::shared_ptr< T > a_out  = std::shared_ptr< T >( new  T[size_out](), //-> init to 0
							    std::default_delete< T[] >() );
	std::shared_ptr< T > z_out  = std::shared_ptr< T >( new  T[size_out](), //-> init to 0
							    std::default_delete< T[] >() );
	std::shared_ptr< T > dz_out = std::shared_ptr< T >( new  T[size_out](), //-> init to 0
							    std::default_delete< T[] >() );
	// compute the activation
	for ( int f = 0 ; f < features_number ; f++ )
	  for ( int s = 0 ; s < size_out ; s++ )
	    a_out.get()[s] += Image_tensors[f][Alps::TensorOrder1::ACTIVATION][s];
	//
	// Compute the feature activation
	for ( int s = 0 ; s < size_out ; s++ )
	  {
	    z_out.get()[s]  = activation_.f(  a_out.get()[s] + *(weights_.get()) );
	    dz_out.get()[s] = activation_.df( a_out.get()[s] + *(weights_.get()) );
	  }

    //
    //
    return std::make_tuple( z_out, dz_out );
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
  class WeightsReconstruction< Type1, Alps::Arch::CUDA, Activation, Solver, Dim > : public Alps::Weights< Type1, Type1 >
  {
    //
    // 
  public:
    /** Constructor. */
    explicit WeightsReconstruction( std::shared_ptr< Alps::Layer >  ){};
    
    /** Destructor */
    virtual ~WeightsReconstruction(){};


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
    virtual std::shared_ptr< Type1 >              get_tensor() const                                   override
    {return weights_;};										      
    // Set size of the tensor									      
    virtual void                                  set_tensor_size( std::vector< std::size_t > )        override{};
    // Set the tensor										      
    virtual void                                  set_tensor( std::shared_ptr< Type1 > )               override{};
												      
    												      
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
    std::shared_ptr< Type1 >           weights_;
    // weights activation
    Activation                         activation_;
    //
    // The mountain observed: fully connected layer
    std::shared_ptr< Alps::Layer >     layer_;
  };
}
#endif
