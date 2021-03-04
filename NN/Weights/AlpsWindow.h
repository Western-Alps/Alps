#ifndef ALPSWINDOW_H
#define ALPSWINDOW_H
//
//
//
#include <iostream>
#include <bits/stdc++.h>
//
#include "MACException.h"
//
//
//
namespace Alps
{
  /** \class Window
   *
   * \brief 
   * Window object represents the basic window element of the convolution.
   * 
   */
  template< class Tensor >
  class Window
    {
      //
      // 
    public:
      /** Constructor. */
      explicit Window(){};
      /** Constructor. */
      explicit Window( const std::vector< int >,
		       const std::vector< int >,
		       const std::vector< int > );
    
      /** Destructor */
      virtual ~Window(){};

      //
      // Accessors

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

      //
      //! tensor
      Tensor tensor_{};

      //! Member representing half convolution window size.
      std::vector< int > w_;
      //! Member for the padding.
      std::vector< int > p_;
      //! Member for the striding.
      std::vector< int > s_;
    };
  //
  //
  template< class T >
  Window<T>::Window( const std::vector< int > Window,
		     const std::vector< int > Padding,
		     const std::vector< int > Striding ):
  w_{Window}, p_{Padding}, s_{Striding}
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
}
#endif
