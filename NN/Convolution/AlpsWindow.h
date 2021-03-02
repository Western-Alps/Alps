#ifndef ALPSWINDOW_H
#define ALPSWINDOW_H
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
  /** \class Window
   *
   * \brief 
   * Window object represents the basic window element of the convolution layer.
   * 
   */
  template< typename Tensor1_Type,
	    typename Tensor2_Type,
	    Alps::Arch Architecture,
	    typename Activation,
	    typename Solver,
	    int Dimension >
  class Window : public Alps::Weights< Tensor1_Type, Tensor2_Type >
    {
      //
      // 
    public:
      /** Constructor. */
      explicit Window( const std::vector< int >,
		       const std::vector< int >,
		       const std::vector< int > ){};
    
      /** Destructor */
      virtual ~Window(){};


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
      virtual void                               forced_update()                                        override{};



    private:
      //
      // private member function
      //! Compare element wise if container1 is bigger than container2
      bool v1_sup_v2_( std::vector< int >::const_iterator First1,
		       std::vector< int >::const_iterator Last1,
		       std::vector< int >::const_iterator First2 )
      {
	while ( First1 != Last1 ) {
	  if ( *First1 < *First2 )
	    return false;
	  ++First1; ++First2;
	}
	return true;
      }
      //! Compare element wise if container1 is less than container2
      bool v1_inf_v2_( std::vector< int >::const_iterator First1,
		       std::vector< int >::const_iterator Last1,
		       std::vector< int >::const_iterator First2 )
      {
	while ( First1 != Last1 ) {
	  if ( !(*First1 > *First2) )
	    return false;
	  ++First1; ++First2;
	}
	return true;
      }

    private:
      //! Matrix of weigths.
      std::shared_ptr< Tensor2_Type >   weights_{nullptr};
      //! Weights activation.
      Activation                        activation_;
      //! The mountain observed: fully connected layer.
      std::shared_ptr< Alps::Layer >    layer_{nullptr};
      //! Member representing half convolution window size.
      std::vector< int > window_;
      //! Member for the padding.
      std::vector< int > padding_;
      //! Member for the striding.
      std::vector< int > striding_;
    };
  /** \class Window
   *
   * \brief 
   * Window object represents the basic window element of the convolution layer.
   * 
   */
  template< typename Type,
	    typename Activation,
	    typename Solver,
	    int Dimension >
  class Window< Type, Eigen::MatrixXd, Alps::Arch::CPU, Activation, Solver, Dimension > : public Alps::Weights< Type, Eigen::MatrixXd >
    {
      //
      // 
    public:
      /** Constructor. */
      explicit Window( std::shared_ptr< Alps::Layer >,
		       const std::vector< int >,
		       const std::vector< int >,
		       const std::vector< int > );
    
      /** Destructor */
      virtual ~Window(){};


    //
    // Accessors
    //
    // Activation tensor from the previous layer
    virtual void set_activations( std::vector< Alps::LayerTensors< Type, 2 > >&,
				  std::vector< Alps::LayerTensors< Type, 2 > >& )                      override{};
    // Get size of the tensor
    virtual const std::vector< std::size_t >       get_tensor_size() const                             override
    { return std::vector< std::size_t >(); };							      
    // Get the tensor										      
    virtual std::shared_ptr< Eigen::MatrixXd >     get_tensor() const                                  override
    {return weights_;};										      
    // Set size of the tensor									      
    virtual void                                   set_tensor_size( std::vector< std::size_t > )       override{};
    // Set the tensor										      
    virtual void                                   set_tensor( std::shared_ptr< Eigen::MatrixXd > )    override{};

    
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
			 std::shared_ptr< Type > > activate( std::vector< Alps::LayerTensors< Type, 2 > >& )       override{};
    // Weighted error
    virtual void                                   weighted_error( std::vector< Alps::LayerTensors< Type, 2 > >&,
								   std::vector< Alps::LayerTensors< Type, 2 > >& ) override{};
    // Update the weights
    virtual void                                   update()                                            override{};
    // Forced the weight update
    virtual void                                   forced_update()                                     override{};




    private:
      //
      // private member function
      //! Compare element wise if container1 is bigger than container2
      bool v1_sup_v2_( std::vector< int >::const_iterator First1,
		       std::vector< int >::const_iterator Last1,
		       std::vector< int >::const_iterator First2 )
      {
	while ( First1 != Last1 ) {
	  if ( *First1 < *First2 )
	    return false;
	  ++First1; ++First2;
	}
	return true;
      }
      //! Compare element wise if container1 is less than container2
      bool v1_inf_v2_( std::vector< int >::const_iterator First1,
		       std::vector< int >::const_iterator Last1,
		       std::vector< int >::const_iterator First2 )
      {
	while ( First1 != Last1 ) {
	  if ( !(*First1 > *First2) )
	    return false;
	  ++First1; ++First2;
	}
	return true;
      }

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
      //! Member representing half convolution window size.
      std::vector< int >                     window_;
      //! Member for the padding.
      std::vector< int >                     padding_;
      //! Member for the striding.
      std::vector< int >                     striding_;
    };
  //
  //
  template< typename T, typename A, typename S, int D >
  Window< T, Eigen::MatrixXd, Alps::Arch::CPU, A, S, D >::Window( std::shared_ptr< Alps::Layer > Layer,
							       const std::vector< int >       Window,
							       const std::vector< int >       Padding,
							       const std::vector< int >       Striding ):
    layer_{Layer}, window_{Window}, padding_{Padding}, striding_{Striding}
  {
    try
      {
//	//
//	// Check the input window dimentions are correct.
//	const int D = tensor_.get_dimension();
//	if ( ( tensor_.get_dimension() != Window.size() ) ||
//	     ( Padding.size()          != Window.size() ) ||
//	     ( Striding.size()         != Window.size() ) )
//	  throw MAC::MACException( __FILE__, __LINE__,
//				   "Miss match dimensions between window and tensor.",
//				   ITK_LOCATION );
//	//
//	// Basic checks on the window dimension.
//	// We do not accept the center of the window outside of the image. Half window should be larger than
//	// padding size.
//	if ( !v1_sup_v2_(Window.begin(), Window.end(), Padding.begin()) )
//	  throw MAC::MACException( __FILE__, __LINE__,
//				   "Half window can't be smaller than the padding.",
//				   ITK_LOCATION );
//	// We do not accept window less than 1 or striding less than 1.
//	//if ( Window < std::vector< int >( D, 1 ) || Striding < std::vector< int >( D, 1 ) )
//	std::vector< int > One( D, 1 );
//	if ( !v1_sup_v2_(Window.begin(),   Window.end(),   One.begin()) ||
//	     !v1_sup_v2_(Striding.begin(), Striding.end(), One.begin()) )
//	  throw MAC::MACException( __FILE__, __LINE__,
//				   "The size of the half window or the striding must be more than 1.",
//				   ITK_LOCATION );
//	// All the window elements must be positive.
//	std::vector< int > Zero( D, 0 );
//	if ( !v1_sup_v2_(Padding.begin(), Padding.end(), Zero.begin()) )
//	  throw MAC::MACException( __FILE__, __LINE__,
//				   "Any window element can't have negative value.",
//				   ITK_LOCATION );
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit(-1);
      }
  };
  /** \class Window
   *
   * \brief 
   * Window object represents the basic window element of the convolution layer.
   * 
   */
  template< typename Type1,
	    typename Type2,
	    typename Activation,
	    typename Solver,
	    int Dimension >
  class Window< Type1, Type2, Alps::Arch::CUDA, Activation, Solver, Dimension > : public Alps::Weights< Type1, Type2 >
    {
      //
      // 
    public:
      /** Constructor. */
      explicit Window( std::shared_ptr< Alps::Layer >,
		       const std::vector< int >,
		       const std::vector< int >,
		       const std::vector< int > ){};
    
      /** Destructor */
      virtual ~Window(){};


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
      //
      // private member function
      //! Compare element wise if container1 is bigger than container2
      bool v1_sup_v2_( std::vector< int >::const_iterator First1,
		       std::vector< int >::const_iterator Last1,
		       std::vector< int >::const_iterator First2 )
      {
	while ( First1 != Last1 ) {
	  if ( *First1 < *First2 )
	    return false;
	  ++First1; ++First2;
	}
	return true;
      }
      //! Compare element wise if container1 is less than container2
      bool v1_inf_v2_( std::vector< int >::const_iterator First1,
		       std::vector< int >::const_iterator Last1,
		       std::vector< int >::const_iterator First2 )
      {
	while ( First1 != Last1 ) {
	  if ( !(*First1 > *First2) )
	    return false;
	  ++First1; ++First2;
	}
	return true;
      }

    private:
      // Matrix of weigths
      std::shared_ptr< Type2 >           weights_;
      // weights activation
      Activation                         activation_;
      //
      // The mountain observed: fully connected layer
      std::shared_ptr< Alps::Layer >     layer_;
      //! Member representing half convolution window size.
      std::vector< int > window_;
      //! Member for the padding.
      std::vector< int > padding_;
      //! Member for the striding.
      std::vector< int > striding_;
    };
}
#endif
