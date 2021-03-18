#ifndef ALPSWINDOW_H
#define ALPSWINDOW_H
//
//
//
#include <iostream>
#include <bits/stdc++.h>
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
// ITK
#include "ITKHeaders.h"
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
  template< class Type, int Dim >
  class Window
    {
      //
      // 
    public:
      /** Constructor. */
      explicit Window( const int,
		       const std::vector< long int >,
		       const std::vector< long int >,
		       const std::vector< long int > );
      /** Destructor */
      virtual ~Window(){};

      //
      // Accessors
      //! get the number of kernel
      const int get_number_kernel() const
      { return k_;};
      //! get  half convolution window size.
      const std::vector< long int >                      get_convolution_window_size() const
      { return w_;};
      //! get the padding.
      const std::vector< long int >                      get_convolution_padding() const
      { return p_;};
      //! get the striding.
      const std::vector< long int >                      get_convolution_striding() const
      { return s_;};
      //! get the number of voxel output per dimension
      const std::array< int, Dim>                        get_output_image_dimensions() const
      { return n_out_; };
      // Sparse matrix holding the index od=f the weights
      const Eigen::SparseMatrix< int, Eigen::RowMajor >  get_weights_matrix() const
      { return weights_matrix_;};
      //! Array with the values of the weights
      const std::vector< std::shared_ptr< Type* > >      get_convolution_weight_values() const
      { return weight_values_;};
      // load image information
      void                                               get_image_information( const typename ImageType< Dim >::RegionType );
      

    private:
      //
      // private member function
      //! Compare element wise if container1 is bigger than container2
      bool v1_sup_v2_( std::vector< long int >::const_iterator First1,
		        std::vector< long int >::const_iterator Last1,
		        std::vector< long int >::const_iterator First2 )
      {
	while ( First1 != Last1 ) {
	  if ( *First1 < *First2 )
	    return false;
	  ++First1; ++First2;
	}
	return true;
      }
      //! Compare element wise if container1 is less than container2
      bool v1_inf_v2_( std::vector< long int >::const_iterator First1,
		        std::vector< long int >::const_iterator Last1,
		        std::vector< long int >::const_iterator First2 )
      {
	while ( First1 != Last1 ) {
	  if ( !(*First1 > *First2) )
	    return false;
	  ++First1; ++First2;
	}
	return true;
      }


      //! Member representing the number of kernel
      int                                          k_;
      //! Member representing half convolution window size.
      std::vector< long int >                      w_;
      //! Member for the padding.
      std::vector< long int >                      p_;
      //! Member for the striding.
      std::vector< long int >                      s_;

      //
      //! Member representing the number of voxel output per dimension
      std::array< int, Dim >                       n_out_;
      // Sparse matrix holding the index od=f the weights
      Eigen::SparseMatrix< int, Eigen::RowMajor >  weights_matrix_;
      //! Array with the values of the weights
      std::vector< std::shared_ptr< Type* > >      weight_values_;
  };
  //
  //
  template< class T, int D >
  Window< T, D >::Window( const int                     Num_kernel,
			  const std::vector< long int > Window,
			  const std::vector< long int > Padding,
			  const std::vector< long int > Striding ):
    k_{Num_kernel}, w_{Window}, p_{Padding}, s_{Striding}
  {
    try
      {
	//
	// Check the number of kernel is not negative
	if ( k_ <= 0 )
	  throw MAC::MACException( __FILE__, __LINE__,
				   "The number of kernels cannot be negative or null.",
				   ITK_LOCATION );
	//
	// Check the input window dimentions are correct.
	if ( D != Window.size() ||
	     D != Window.size() ||
	     D != Window.size() )
	  throw MAC::MACException( __FILE__, __LINE__,
				   "Mismatch dimensions between the window and the dimension asked.",
				   ITK_LOCATION );
	//
	// Basic checks on the window dimension.
	// We do not accept the center of the window outside of the image. Half window should be larger than
	// padding size.
	if ( !v1_sup_v2_(Window.begin(), Window.end(), Padding.begin()) )
	  throw MAC::MACException( __FILE__, __LINE__,
				   "Half window can't be smaller than the padding.",
				   ITK_LOCATION );
	// We do not accept window less than 1 or striding less than 1.
	//if ( Window < std::vector< int >( D, 1 ) || Striding < std::vector< int >( D, 1 ) )
	std::vector< long int > One( D, 1 );
	if ( !v1_sup_v2_(Striding.begin(), Striding.end(), One.begin()) )
	  throw MAC::MACException( __FILE__, __LINE__,
				   "The size of the striding must be more than 1.",
				   ITK_LOCATION );
	// All the window elements must be positive.
	std::vector< long int > Zero( D, 0 );
	if ( !v1_sup_v2_(Padding.begin(), Padding.end(), Zero.begin()) ||
	     !v1_sup_v2_(Window.begin(),  Window.end(),  Zero.begin()) )
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
  //
  //
  template< class T, int D > void
  Window< T, D >::get_image_information( const typename ImageType< D >::RegionType Region )
  {
    try
      {
	//
	//
	typename ImageType< D >::SizeType size = Region.GetSize();
	//
	int
	  rows = 1,
	  cols = 1 ;
	for ( long int d = 0 ; d < D ; d++ )
	  {
	    if ( static_cast<long int>(size[d]) > 2 * (w_[d] - p_[d]) )
	      n_out_[d] = ( static_cast<long int>(size[d]) - 2 * (w_[d] - p_[d]) ) / s_[d];
	    else
	      throw MAC::MACException( __FILE__, __LINE__,
				       "The window dimensions exceed the input image size.",
				       ITK_LOCATION );
	    //
	    rows *= n_out_[d];
	    cols *= size[d];
 	  }
	//
	weights_matrix_.resize( rows, cols );

	
	////////////////////////////////
	//                            //
	// Mapping of the input image //
	//                            //
	////////////////////////////////
	
	
	//
	//
	switch( D )
	  {
	  case 0:
	    {
	      throw MAC::MACException( __FILE__, __LINE__,
				       "Building convolution windows with null dimension as not been thought yet.",
				       ITK_LOCATION );
	      break;
	    }
	  case 1:
	    {
	      for ( long int x0 = (w_[0] - p_[0]) ; x0 < static_cast< long int >(size[0]) ; x0++ )
		{}
	      break;
	    }
	  case 2:
	    {
	      int
		row = 0;
	      for ( long int x1 = (w_[1] - p_[1]) ; x1 < static_cast< long int >(size[1]) - (w_[1] - p_[1]) ; x1 = x1 + s_[1] )
		for ( long int x0 = (w_[0] - p_[0]) ; x0 < static_cast< long int >(size[0]) - (w_[0] - p_[0]) ; x0 = x0 + s_[0] )
		  {
		    //
		    // window
		    
		    long int
		      widx   = 1, // index of the weight; 0 will be the bias
		      idx    = 0, // index from the image
		      pos_x0 = 0, pos_x1 = 0;
		    //
		    for ( long int w1 = -w_[1] ; w1 < w_[1]+1 ; w1++ )
		      for ( long int w0 = -w_[0] ; w0 < w_[0]+1 ; w0++ )
			{
			  // we check the position are in the image
			  pos_x0 = x0 + w0;
			  pos_x1 = x1 + w1;
			  if ( pos_x0 >= 0 && pos_x1 >= 0 &&
			       pos_x0 < static_cast< long int >(size[0]) &&
			       pos_x1 < static_cast< long int >(size[1]) )
			    {
			      idx = (x0 + w0) + static_cast< long int >(size[0]) * (x1 + w1);
			      weights_matrix_.insert( row, idx ) = widx++;
			    }
			}
		    //
		    row++;
		  }
	      //
	      break;
	    }
	  case 3:
	  case 4:
	  case 5:
	  case 6:
	  default:
	    {
	      throw MAC::MACException( __FILE__, __LINE__,
				       "Building convolution windows with high dimensions as not been thought yet, but can be customized.",
				       ITK_LOCATION );
	      break;
	    }
	  }
	
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit(-1);
      }
  };
}
#endif
