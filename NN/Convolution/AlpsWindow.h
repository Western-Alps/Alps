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
	//
	// Check the input window dimentions are correct.
	const int D = tensor_.get_dimension();
	if ( ( tensor_.get_dimension() != Window.size() ) ||
	     ( Padding.size()          != Window.size() ) ||
	     ( Striding.size()         != Window.size() ) )
	  throw MAC::MACException( __FILE__, __LINE__,
				   "Miss match dimensions between window and tensor.",
				   ITK_LOCATION );
	//
	// Basic chacks on the window dimension.
	// We do not accept the center of the window outside of the image.
	if ( Window < Padding )
	  throw MAC::MACException( __FILE__, __LINE__,
				   "Half window can't be smaller than the padding.",
				   ITK_LOCATION );
	// We do not accept window less than 1 or striding less than 1.
	if ( Window < std::vector< int >( D, 1 ) || Striding < std::vector< int >( D, 1 ) )
	  throw MAC::MACException( __FILE__, __LINE__,
				   "The size of the half window or the striding must be more than 1.",
				   ITK_LOCATION );
	// All the window elements must be positive.
	if ( Window   < std::vector< int >( D, 0 ) ||
	     Padding  < std::vector< int >( D, 0 ) ||
	     Striding < std::vector< int >( D, 0 ) )
	  throw MAC::MACException( __FILE__, __LINE__,
				   "Any window element can't have negative value.",
				   ITK_LOCATION );
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit(-1);
      }
  };
}
#endif
