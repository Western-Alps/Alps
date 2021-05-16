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
      const int                                          get_number_kernel() const
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
      const std::array< std::size_t, Dim>                get_output_image_dimensions() const
      { return n_out_; };
      //! Sparse matrix holding the index od=f the weights
      const Eigen::SparseMatrix< int, Eigen::RowMajor >& get_weights_matrix() const
      { return weights_matrix_;};
      //! Array with the values of the weights
      std::shared_ptr< Type >&                           get_convolution_weight_values( const int Kernel ) const
      { return weight_values_[Kernel];};
      //! Array with the derivative values of the weights
      std::vecto< std::shared_ptr< Type > >              get_derivated_weight_values() const
      { return derivated_weight_values_;};
      //! load image information
      void                                               get_image_information( const typename ImageType< Dim >::RegionType );
      //
      //
      //! Set array with the values of the weights
      void                                               set_convolution_weight_values( const int Kernel, W ) 
      { weight_values_[Kernel] = W;};


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
      std::array< std::size_t, Dim >               n_out_;
      //! Sparse matrix holding the index od=f the weights
      Eigen::SparseMatrix< int, Eigen::RowMajor >  weights_matrix_;
      //! Array with the values of the weights
      std::vector< std::shared_ptr< Type > >       weight_values_;
      //! Array with the values of the weights
      std::vector< std::shared_ptr< Type > >       derivated_weight_values_;
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


	///////////////////////
	// Build the kernels //
	///////////////////////
	//
	// resize to the number of kernel requiered
	weight_values_.resize( Num_kernel );
	long int number_weights = 1;
	// size of the kernel
	for ( int d = 0 ; d < D ; d++ )
	  number_weights *= 2 * Window[d] + 1;
	// add the bias
	number_weights += 1;
	//
	// set the kernel values 
	std::default_random_engine          generator;
	std::uniform_real_distribution< T > distribution( -1.0, 1.0 );
	// feel up the arrays
	for ( int k = 0 ; k < Num_kernel ; k++ )
	  {
	    weight_values_[k] = std::shared_ptr< T >( new T[number_weights],
						      std::default_delete< T[] >() );
	    //
	    for ( int w = 0 ; w < number_weights ; w++ )
	      {
		weight_values_[k].get()[w] = distribution( generator );
//		//
//		std::cout << "weight_values_[kernel: "
//			  <<k<< "].get()[weight: "
//			  <<w<<"] = "
//			  << weight_values_[k].get()[w]
//			  << std::endl;
	      }
	  }

	////////////////////////////////
	// Build the derivated values //
	////////////////////////////////
	//
	derivated_weight_values_.resize( number_weights );
	for ( int w = 1 ; w < number_weights ; w++ )
	  {
	    derivated_weight_values_[w] = std::shared_ptr< T >( new T[number_weights](),
								std::default_delete< T[] >() );
	    derivated_weight_values_[w].get()[w] = 1.;
	  }
	// The bias
	derivated_weight_values_[0] = std::shared_ptr< T >( new T[number_weights](),
							    std::default_delete< T[] >() );
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
	      n_out_[d] = ( static_cast< std::size_t >(size[d]) - 2 * (w_[d] - p_[d]) ) / s_[d];
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
	      int row = 0;
	      for ( long int x0 = (w_[0] - p_[0]) ; x0 < static_cast< long int >(size[0]) - (w_[0] - p_[0]) ; x0 = x0 + s_[0] )
		{
		  //
		  // window
		  long int
		    widx   = 1, // index of the weight; 0 will be the bias
		    idx    = 0, // index from the image
		    pos_x0 = 0;
		  //
		  for ( long int w0 = -w_[0] ; w0 < w_[0]+1 ; w0++ )
		    {
		      // we check the position are in the image
		      pos_x0 = x0 + w0;
		      if ( pos_x0 >= 0 &&
			   pos_x0 < static_cast< long int >(size[0]) )
			{
			  idx = (x0 + w0);
			  weights_matrix_.insert( row, idx ) = widx++;
			}
		    }
		  //
		  row++;
		}
	      //
	      break;
	    }
	  case 2:
	    {
	      int row = 0;
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
	    {
	      int row = 0;
	      for ( long int x2 = (w_[2] - p_[2]) ; x2 < static_cast< long int >(size[2]) - (w_[2] - p_[2]) ; x2 = x2 + s_[2] )
		for ( long int x1 = (w_[1] - p_[1]) ; x1 < static_cast< long int >(size[1]) - (w_[1] - p_[1]) ; x1 = x1 + s_[1] )
		  for ( long int x0 = (w_[0] - p_[0]) ; x0 < static_cast< long int >(size[0]) - (w_[0] - p_[0]) ; x0 = x0 + s_[0] )
		    {
		      //
		      // window
		      long int
			widx   = 1, // index of the weight; 0 will be the bias
			idx    = 0, // index from the image
			pos_x0 = 0, pos_x1 = 0, pos_x2 = 0;
		      //
		      for ( long int w2 = -w_[2] ; w2 < w_[2]+1 ; w2++ )
			for ( long int w1 = -w_[1] ; w1 < w_[1]+1 ; w1++ )
			  for ( long int w0 = -w_[0] ; w0 < w_[0]+1 ; w0++ )
			    {
			      // we check the position are in the image
			      pos_x0 = x0 + w0;
			      pos_x1 = x1 + w1;
			      pos_x2 = x2 + w2;
			      if ( pos_x0 >= 0 && pos_x1 >= 0 && pos_x2 >= 0 &&
				   pos_x0 < static_cast< long int >(size[0]) &&
				   pos_x1 < static_cast< long int >(size[1]) &&
				   pos_x2 < static_cast< long int >(size[2]) )
				{
				  idx = (x0 + w0) + static_cast< long int >(size[0]) * (x1 + w1) + static_cast< long int >(size[0]) * static_cast< long int >(size[1]) * (x2 + w2);
				  weights_matrix_.insert( row, idx ) = widx++;
				}
			    }
		      //
		      row++;
		    }
	      //
	      break;
	    }
	  case 4:
	    {
	      int row = 0;
	      for ( long int x3 = (w_[3] - p_[3]) ; x3 < static_cast< long int >(size[3]) - (w_[3] - p_[3]) ; x3 = x3 + s_[3] )
		for ( long int x2 = (w_[2] - p_[2]) ; x2 < static_cast< long int >(size[2]) - (w_[2] - p_[2]) ; x2 = x2 + s_[2] )
		  for ( long int x1 = (w_[1] - p_[1]) ; x1 < static_cast< long int >(size[1]) - (w_[1] - p_[1]) ; x1 = x1 + s_[1] )
		    for ( long int x0 = (w_[0] - p_[0]) ; x0 < static_cast< long int >(size[0]) - (w_[0] - p_[0]) ; x0 = x0 + s_[0] )
		      {
			//
			// window
			long int
			  widx   = 1, // index of the weight; 0 will be the bias
			  idx    = 0, // index from the image
			  pos_x0 = 0, pos_x1 = 0, pos_x2 = 0, pos_x3 = 0;
			//
			for ( long int w3 = -w_[3] ; w3 < w_[3]+1 ; w3++ )
			  for ( long int w2 = -w_[2] ; w2 < w_[2]+1 ; w2++ )
			    for ( long int w1 = -w_[1] ; w1 < w_[1]+1 ; w1++ )
			      for ( long int w0 = -w_[0] ; w0 < w_[0]+1 ; w0++ )
				{
				  // we check the position are in the image
				  pos_x0 = x0 + w0;
				  pos_x1 = x1 + w1;
				  pos_x2 = x2 + w2;
				  pos_x3 = x3 + w3;
				  if ( pos_x0 >= 0 && pos_x1 >= 0 && pos_x2 >= 0 &&
				       pos_x3 >= 0 &&
				       pos_x0 < static_cast< long int >(size[0]) &&
				       pos_x1 < static_cast< long int >(size[1]) &&
				       pos_x2 < static_cast< long int >(size[2]) &&
				       pos_x3 < static_cast< long int >(size[3]) )
				    {
				      idx  = (x0 + w0) + static_cast< long int >(size[0]) * (x1 + w1) + static_cast< long int >(size[0]) * static_cast< long int >(size[1]) * (x2 + w2);
				      idx +=  static_cast< long int >(size[0]) * static_cast< long int >(size[1]) * static_cast< long int >(size[2]) * (x3 + w3);
				      weights_matrix_.insert( row, idx ) = widx++;
				    }
				}
			//
			row++;
		      }
	      //
	      break;
	    }
	  case 5:
	    {
	      int row = 0;
	      for ( long int x4 = (w_[4] - p_[4]) ; x4 < static_cast< long int >(size[4]) - (w_[4] - p_[4]) ; x4 = x4 + s_[4] )
		for ( long int x3 = (w_[3] - p_[3]) ; x3 < static_cast< long int >(size[3]) - (w_[3] - p_[3]) ; x3 = x3 + s_[3] )
		  for ( long int x2 = (w_[2] - p_[2]) ; x2 < static_cast< long int >(size[2]) - (w_[2] - p_[2]) ; x2 = x2 + s_[2] )
		    for ( long int x1 = (w_[1] - p_[1]) ; x1 < static_cast< long int >(size[1]) - (w_[1] - p_[1]) ; x1 = x1 + s_[1] )
		      for ( long int x0 = (w_[0] - p_[0]) ; x0 < static_cast< long int >(size[0]) - (w_[0] - p_[0]) ; x0 = x0 + s_[0] )
			{
			  //
			  // window
			  long int
			    widx   = 1, // index of the weight; 0 will be the bias
			    idx    = 0, // index from the image
			    pos_x0 = 0, pos_x1 = 0, pos_x2 = 0, pos_x3 = 0, pos_x4 = 0;
			  //
			  for ( long int w4 = -w_[4] ; w4 < w_[4]+1 ; w4++ )
			    for ( long int w3 = -w_[3] ; w3 < w_[3]+1 ; w3++ )
			      for ( long int w2 = -w_[2] ; w2 < w_[2]+1 ; w2++ )
				for ( long int w1 = -w_[1] ; w1 < w_[1]+1 ; w1++ )
				  for ( long int w0 = -w_[0] ; w0 < w_[0]+1 ; w0++ )
				    {
				      // we check the position are in the image
				      pos_x0 = x0 + w0;
				      pos_x1 = x1 + w1;
				      pos_x2 = x2 + w2;
				      pos_x3 = x3 + w3;
				      pos_x4 = x4 + w4;
				      if ( pos_x0 >= 0 && pos_x1 >= 0 && pos_x2 >= 0 &&
					   pos_x3 >= 0 && pos_x4 >= 0 &&
					   pos_x0 < static_cast< long int >(size[0]) &&
					   pos_x1 < static_cast< long int >(size[1]) &&
					   pos_x2 < static_cast< long int >(size[2]) &&
					   pos_x3 < static_cast< long int >(size[3]) &&
					   pos_x4 < static_cast< long int >(size[4]) )
					{
					  idx  = (x0 + w0) + static_cast< long int >(size[0]) * (x1 + w1) + static_cast< long int >(size[0]) * static_cast< long int >(size[1]) * (x2 + w2);
					  idx +=  static_cast< long int >(size[0]) * static_cast< long int >(size[1]) * static_cast< long int >(size[2]) * (x3 + w3);
					  idx +=  static_cast< long int >(size[0]) * static_cast< long int >(size[1]) * static_cast< long int >(size[2]) * static_cast< long int >(size[3]) * (x4 + w4);
					  weights_matrix_.insert( row, idx ) = widx++;
					}
				    }
			//
			row++;
		      }
	      //
	      break;
	    }
	  case 6:
	    {
	      int row = 0;
	      for ( long int x5 = (w_[5] - p_[5]) ; x5 < static_cast< long int >(size[5]) - (w_[5] - p_[5]) ; x5 = x5 + s_[5] )
		for ( long int x4 = (w_[4] - p_[4]) ; x4 < static_cast< long int >(size[4]) - (w_[4] - p_[4]) ; x4 = x4 + s_[4] )
		  for ( long int x3 = (w_[3] - p_[3]) ; x3 < static_cast< long int >(size[3]) - (w_[3] - p_[3]) ; x3 = x3 + s_[3] )
		    for ( long int x2 = (w_[2] - p_[2]) ; x2 < static_cast< long int >(size[2]) - (w_[2] - p_[2]) ; x2 = x2 + s_[2] )
		      for ( long int x1 = (w_[1] - p_[1]) ; x1 < static_cast< long int >(size[1]) - (w_[1] - p_[1]) ; x1 = x1 + s_[1] )
			for ( long int x0 = (w_[0] - p_[0]) ; x0 < static_cast< long int >(size[0]) - (w_[0] - p_[0]) ; x0 = x0 + s_[0] )
			  {
			    //
			    // window
			    long int
			      widx   = 1, // index of the weight; 0 will be the bias
			      idx    = 0, // index from the image
			      pos_x0 = 0, pos_x1 = 0, pos_x2 = 0, pos_x3 = 0, pos_x4 = 0, pos_x5 = 0;
			    //
			    for ( long int w5 = -w_[5] ; w5 < w_[5]+1 ; w5++ )
			      for ( long int w4 = -w_[4] ; w4 < w_[4]+1 ; w4++ )
				for ( long int w3 = -w_[3] ; w3 < w_[3]+1 ; w3++ )
				  for ( long int w2 = -w_[2] ; w2 < w_[2]+1 ; w2++ )
				    for ( long int w1 = -w_[1] ; w1 < w_[1]+1 ; w1++ )
				      for ( long int w0 = -w_[0] ; w0 < w_[0]+1 ; w0++ )
					{
					  // we check the position are in the image
					  pos_x0 = x0 + w0;
					  pos_x1 = x1 + w1;
					  pos_x2 = x2 + w2;
					  pos_x3 = x3 + w3;
					  pos_x4 = x4 + w4;
					  pos_x5 = x5 + w5;
					  if ( pos_x0 >= 0 && pos_x1 >= 0 && pos_x2 >= 0 &&
					       pos_x3 >= 0 && pos_x4 >= 0 && pos_x5 >= 0 &&
					       pos_x0 < static_cast< long int >(size[0]) &&
					       pos_x1 < static_cast< long int >(size[1]) &&
					       pos_x2 < static_cast< long int >(size[2]) &&
					       pos_x3 < static_cast< long int >(size[3]) &&
					       pos_x4 < static_cast< long int >(size[4]) &&
					       pos_x5 < static_cast< long int >(size[5]) )
					    {
					      idx  = (x0 + w0) + static_cast< long int >(size[0]) * (x1 + w1) + static_cast< long int >(size[0]) * static_cast< long int >(size[1]) * (x2 + w2);
					      idx +=  static_cast< long int >(size[0]) * static_cast< long int >(size[1]) * static_cast< long int >(size[2]) * (x3 + w3);
					      idx +=  static_cast< long int >(size[0]) * static_cast< long int >(size[1]) * static_cast< long int >(size[2]) * static_cast< long int >(size[3]) * (x4 + w4);
					      idx +=  static_cast< long int >(size[0]) * static_cast< long int >(size[1]) * static_cast< long int >(size[2]) * static_cast< long int >(size[3]) *
						static_cast< long int >(size[4]) * (x5 + w5);
					      weights_matrix_.insert( row, idx ) = widx++;
					    }
					}
			    //
			    row++;
			  }
	      //
	      break;
	    }
	  case 7:
	  case 8:
	  case 9:
	  default:
	    {
	      std::string mess = "Building convolution windows with high dimensions as not been thought yet, but can be customized.\n";
	      mess += "It is also important to note that ITK might not allowd very high dimensions images.";
	      throw MAC::MACException( __FILE__, __LINE__,
				       mess.c_str(),
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
