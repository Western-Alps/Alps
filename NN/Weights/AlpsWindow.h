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
#ifndef ALPSWINDOW_H
#define ALPSWINDOW_H
//
//
//
#include <iostream>
#include <bits/stdc++.h>
#include <algorithm>    // std::transform
#include <functional>   // std::plus
#include <math.h>
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
      const int                                          get_number_kernel() const noexcept
      { return k_;};
      //! get  half convolution window size.
      const std::vector< long int >                      get_convolution_window_size() const noexcept
      { return w_;};
      //! get the padding.
      const std::vector< long int >                      get_convolution_padding() const noexcept
      { return p_;};
      //! get the striding.
      const std::vector< long int >                      get_convolution_striding() const noexcept
      { return s_;};
      //! get the number of voxel output per dimension
      const std::array< std::size_t, Dim>                get_output_image_dimensions() const noexcept
      { return (transposed_ ? n_in_ : n_out_); };
      //! Sparse matrix holding the index od=f the weights
      const Eigen::SparseMatrix< int, Eigen::RowMajor >& get_weights_matrix() const 
      {return weights_matrix_;};
      //! Array with the values of the weights
      const std::vector< Type >&                         get_convolution_weight_values( const int Kernel ) const
      { return weight_values_[Kernel];};
      //! Array with the derivative values of the weights
      const std::vector< std::vector< Type > >&          get_derivated_weight_values() const
      { return derivated_weight_values_;};
      //! load image information
      void                                               get_image_information( const typename ImageType< Dim >::RegionType );
      //
      //
      //! Set array with the values of the weights
      void                                               set_convolution_weight_values( const int Kernel,
											std::vector< Type > W )
      { std::transform( weight_values_[Kernel].begin(), weight_values_[Kernel].end(),
			W.begin(), weight_values_[Kernel].begin(), std::plus< Type >());};
      //! Set if the window is used as transposed of not
      void                                               set_transpose( const bool Transpose )
      { transposed_ = Transpose;};
      //! Set if the window is used as transposed of not
      const bool                                         get_transpose() const
      { return transposed_;};


      //
      // Functions
      const bool                                         initialized() const
      { return (weights_matrix_.nonZeros() == 0 ? false : true );};
      // Save the weights at the end of the epoque
      void                                               save_weights( std::ofstream& ) const;


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
      //! Member representing the number or weights and the bias in one window
      int                                          number_weights_{1};
      //! Member representing half convolution window size.
      std::vector< long int >                      w_;
      //! Member for the padding.
      std::vector< long int >                      p_;
      //! Member for the striding.
      std::vector< long int >                      s_;

      //
      //! Member checking if we are using the direct window or the transposed
      bool                                         transposed_{false};
      //! Member representing the number of voxel input per dimension
      std::array< std::size_t, Dim >               n_in_;
      //! Member representing the number of voxel output per dimension
      std::array< std::size_t, Dim >               n_out_;
      //! Sparse matrix holding the index od=f the weights
      Eigen::SparseMatrix< int, Eigen::RowMajor >  weights_matrix_;
      //! Array with the values of the weights
      std::vector< std::vector< Type > >           weight_values_;
      //! Array with the values of the weights
      std::vector< std::vector< Type > >           derivated_weight_values_;
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
	// Check the input window dimensions are correct.
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
	// size of the kernel
	for ( int d = 0 ; d < D ; d++ )
	  number_weights_ *= 2 * Window[d] + 1;
	// add the bias
	number_weights_ += 1;


	////////////////////////////////
	// Build the derivated values //
	////////////////////////////////
	//
	derivated_weight_values_.resize( number_weights_ );
	for ( int w = 1 ; w < number_weights_ ; w++ )
	  {
	    derivated_weight_values_[w] = std::vector< T >( number_weights_, 0. );
	    // ToDo: there is something weird here!
	    derivated_weight_values_[w][w] = 1.;
	  }
	// The bias
	derivated_weight_values_[0] = std::vector< T >( number_weights_, 0. );
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
	  if ( transposed_ )
	    {
	      // record the output dimensions
	      n_out_[d] = size[d];
	      // creates the input dimensions
	      n_in_[d] = static_cast< std::size_t >( floor(size[d] * s_[d] + 2 * (w_[d] - p_[d])) );
	      if ( static_cast<long int>(n_in_[d]) <= 2 * (w_[d] - p_[d]) )
		{
		  std::cout << "size[" << d <<"] = " << size[d] << std::endl;
		  std::cout << "w_[d] = " << w_[d] << std::endl;
		  std::cout << "p_[d] = " << p_[d] << std::endl;
		  std::cout << "2 * (w_[d] - p_[d]) = " << 2 * (w_[d] - p_[d]) << std::endl;
		  throw MAC::MACException( __FILE__, __LINE__,
					   "The window dimensions exceed the input image size.",
					   ITK_LOCATION );
		}
	      //
	      rows *= n_out_[d];
	      cols *= n_in_[d];
	    }
	  else
	    {
	      // record the input dimensions
	      n_in_[d] = size[d];
	      // creates the output dimensions
	      if ( static_cast<long int>(size[d]) > 2 * (w_[d] - p_[d]) )
		n_out_[d] = static_cast< std::size_t >( ceil((size[d] - 2 * (w_[d] - p_[d])) / s_[d]) );
	      else
		{
		  std::cout << "size[" << d <<"] = " << size[d] << std::endl;
		  std::cout << "w_[d] = " << w_[d] << std::endl;
		  std::cout << "p_[d] = " << p_[d] << std::endl;
		  std::cout << "2 * (w_[d] - p_[d]) = " << 2 * (w_[d] - p_[d]) << std::endl;
		  throw MAC::MACException( __FILE__, __LINE__,
					   "The window dimensions exceed the input image size.",
					   ITK_LOCATION );
		}
	      //
	      rows *= n_out_[d];
	      cols *= n_in_[d];
	    }
	    
	//
	// Information on the convolution matrix
	{
	  //
	  // Input and output images
	  std::string
	    size_in_str = "[ ",
	    size_out_str = "[ ",
	    Conv_matrix_str = " The dimension of the convolution matrix is [";
	  //
	  for ( long int d = 0 ; d < D ; d++ )
	    if ( transposed_ )
	      {
		size_in_str  += std::to_string( n_out_[d]  ) + " ";
		size_out_str += std::to_string( n_in_[d] ) + " ";
	      }
	    else
	      {
		size_in_str  += std::to_string( n_in_[d]  ) + " ";
		size_out_str += std::to_string( n_out_[d] ) + " ";
	      }
	  //
	  size_in_str  += "]";
	  size_out_str += "]";
	  //
	  //
	  std::string mess = "The dimension of the input image is: " + size_in_str + ".";
	  //
	  mess += " The dimention of the output image is: " + size_out_str + ".";
	  mess += Conv_matrix_str + std::to_string(rows) + "x" + std::to_string(cols) + "].";
	  //
	  std::cout << mess << std::endl;
	}
	//
	weights_matrix_.resize( rows, cols );

	
	///////////////////////
	// Build the kernels //
	///////////////////////
	//
	// set the kernel values
	double                              limit = std::sqrt( 6. / (rows+cols) );
	std::default_random_engine          generator;
	std::uniform_real_distribution< T > distribution( -limit, limit );
	// feel up the arrays
	for ( int k = 0 ; k < k_ ; k++ )
	  {
	    weight_values_[k] = std::vector< T >( number_weights_, 0. );
	    // we live the bias at zero
	    for ( int w = 1 ; w < number_weights_ ; w++ )
	      {
		weight_values_[k][w] = distribution( generator );
//		//
//		std::cout << "weight_values_[kernel: "
//			  <<k<< "].get()[weight: "
//			  <<w<<"] = "
//			  << weight_values_[k][w]
//			  << std::endl;
	      }
	  }

	////////////////////////////////
	//                            //
	// Mapping of the input image //
	//                            //
	////////////////////////////////
	
	
	//
	// This part of the program maps the weights index of each weight in the flatten image.
	// It should look like the follwing matrix, where 1,2,3, ... are the indexes of the
	// weight 1, 2, 3, ...
	// 1 2 3 0 4 5 6 0 7 8 9 0 0 0 0 0 
	// 0 1 2 3 0 4 5 6 0 7 8 9 0 0 0 0 
	// 0 0 0 0 1 2 3 0 4 5 6 0 7 8 9 0 
	// 0 0 0 0 0 1 2 3 0 4 5 6 0 7 8 9 
	//
	switch( D )
	  {
	  case 0:
	    {
	      throw MAC::MACException( __FILE__, __LINE__,
				       "Building convolution windows with null dimension has not been thought yet.",
				       ITK_LOCATION );
	      break;
	    }
	  case 1:
	    {
	      int row = 0;
	      for ( long int x0 = (w_[0] - p_[0]) ; x0 < static_cast< long int >(n_in_[0]) - (w_[0] - p_[0]) ; x0 = x0 + s_[0] )
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
			   pos_x0 < static_cast< long int >(n_in_[0]) )
			{
			  idx = (x0 + w0);
			  weights_matrix_.insert( row, idx ) = widx++;
			}
		      else
			widx++;
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
	      for ( long int x1 = (w_[1] - p_[1]) ; x1 < static_cast< long int >(n_in_[1]) - (w_[1] - p_[1]) ; x1 = x1 + s_[1] )
		for ( long int x0 = (w_[0] - p_[0]) ; x0 < static_cast< long int >(n_in_[0]) - (w_[0] - p_[0]) ; x0 = x0 + s_[0] )
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
			       pos_x0 < static_cast< long int >(n_in_[0]) &&
			       pos_x1 < static_cast< long int >(n_in_[1]) )
			    {
			      idx = (x0 + w0) + static_cast< long int >(n_in_[0]) * (x1 + w1);
			      weights_matrix_.insert( row, idx ) = widx++;
			    }
			  else
			    widx++;
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
	      for ( long int x2 = (w_[2] - p_[2]) ; x2 < static_cast< long int >(n_in_[2]) - (w_[2] - p_[2]) ; x2 = x2 + s_[2] )
		for ( long int x1 = (w_[1] - p_[1]) ; x1 < static_cast< long int >(n_in_[1]) - (w_[1] - p_[1]) ; x1 = x1 + s_[1] )
		  for ( long int x0 = (w_[0] - p_[0]) ; x0 < static_cast< long int >(n_in_[0]) - (w_[0] - p_[0]) ; x0 = x0 + s_[0] )
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
				   pos_x0 < static_cast< long int >(n_in_[0]) &&
				   pos_x1 < static_cast< long int >(n_in_[1]) &&
				   pos_x2 < static_cast< long int >(n_in_[2]) )
				{
				  idx = (x0 + w0) + static_cast< long int >(n_in_[0]) * (x1 + w1) + static_cast< long int >(n_in_[0]) * static_cast< long int >(n_in_[1]) * (x2 + w2);
				  weights_matrix_.insert( row, idx ) = widx++;
				}
			      else
				widx++;
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
	      for ( long int x3 = (w_[3] - p_[3]) ; x3 < static_cast< long int >(n_in_[3]) - (w_[3] - p_[3]) ; x3 = x3 + s_[3] )
		for ( long int x2 = (w_[2] - p_[2]) ; x2 < static_cast< long int >(n_in_[2]) - (w_[2] - p_[2]) ; x2 = x2 + s_[2] )
		  for ( long int x1 = (w_[1] - p_[1]) ; x1 < static_cast< long int >(n_in_[1]) - (w_[1] - p_[1]) ; x1 = x1 + s_[1] )
		    for ( long int x0 = (w_[0] - p_[0]) ; x0 < static_cast< long int >(n_in_[0]) - (w_[0] - p_[0]) ; x0 = x0 + s_[0] )
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
				       pos_x0 < static_cast< long int >(n_in_[0]) &&
				       pos_x1 < static_cast< long int >(n_in_[1]) &&
				       pos_x2 < static_cast< long int >(n_in_[2]) &&
				       pos_x3 < static_cast< long int >(n_in_[3]) )
				    {
				      idx  = (x0 + w0) + static_cast< long int >(n_in_[0]) * (x1 + w1) + static_cast< long int >(n_in_[0]) * static_cast< long int >(n_in_[1]) * (x2 + w2);
				      idx +=  static_cast< long int >(n_in_[0]) * static_cast< long int >(n_in_[1]) * static_cast< long int >(n_in_[2]) * (x3 + w3);
				      weights_matrix_.insert( row, idx ) = widx++;
				    }
				  else
				    widx++;
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
	      for ( long int x4 = (w_[4] - p_[4]) ; x4 < static_cast< long int >(n_in_[4]) - (w_[4] - p_[4]) ; x4 = x4 + s_[4] )
		for ( long int x3 = (w_[3] - p_[3]) ; x3 < static_cast< long int >(n_in_[3]) - (w_[3] - p_[3]) ; x3 = x3 + s_[3] )
		  for ( long int x2 = (w_[2] - p_[2]) ; x2 < static_cast< long int >(n_in_[2]) - (w_[2] - p_[2]) ; x2 = x2 + s_[2] )
		    for ( long int x1 = (w_[1] - p_[1]) ; x1 < static_cast< long int >(n_in_[1]) - (w_[1] - p_[1]) ; x1 = x1 + s_[1] )
		      for ( long int x0 = (w_[0] - p_[0]) ; x0 < static_cast< long int >(n_in_[0]) - (w_[0] - p_[0]) ; x0 = x0 + s_[0] )
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
					   pos_x0 < static_cast< long int >(n_in_[0]) &&
					   pos_x1 < static_cast< long int >(n_in_[1]) &&
					   pos_x2 < static_cast< long int >(n_in_[2]) &&
					   pos_x3 < static_cast< long int >(n_in_[3]) &&
					   pos_x4 < static_cast< long int >(n_in_[4]) )
					{
					  idx  = (x0 + w0) + static_cast< long int >(n_in_[0]) * (x1 + w1) + static_cast< long int >(n_in_[0]) * static_cast< long int >(n_in_[1]) * (x2 + w2);
					  idx +=  static_cast< long int >(n_in_[0]) * static_cast< long int >(n_in_[1]) * static_cast< long int >(n_in_[2]) * (x3 + w3);
					  idx +=  static_cast< long int >(n_in_[0]) * static_cast< long int >(n_in_[1]) * static_cast< long int >(n_in_[2]) * static_cast< long int >(n_in_[3]) * (x4 + w4);
					  weights_matrix_.insert( row, idx ) = widx++;
					}
				      else
					widx++;
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
	      for ( long int x5 = (w_[5] - p_[5]) ; x5 < static_cast< long int >(n_in_[5]) - (w_[5] - p_[5]) ; x5 = x5 + s_[5] )
		for ( long int x4 = (w_[4] - p_[4]) ; x4 < static_cast< long int >(n_in_[4]) - (w_[4] - p_[4]) ; x4 = x4 + s_[4] )
		  for ( long int x3 = (w_[3] - p_[3]) ; x3 < static_cast< long int >(n_in_[3]) - (w_[3] - p_[3]) ; x3 = x3 + s_[3] )
		    for ( long int x2 = (w_[2] - p_[2]) ; x2 < static_cast< long int >(n_in_[2]) - (w_[2] - p_[2]) ; x2 = x2 + s_[2] )
		      for ( long int x1 = (w_[1] - p_[1]) ; x1 < static_cast< long int >(n_in_[1]) - (w_[1] - p_[1]) ; x1 = x1 + s_[1] )
			for ( long int x0 = (w_[0] - p_[0]) ; x0 < static_cast< long int >(n_in_[0]) - (w_[0] - p_[0]) ; x0 = x0 + s_[0] )
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
					       pos_x0 < static_cast< long int >(n_in_[0]) &&
					       pos_x1 < static_cast< long int >(n_in_[1]) &&
					       pos_x2 < static_cast< long int >(n_in_[2]) &&
					       pos_x3 < static_cast< long int >(n_in_[3]) &&
					       pos_x4 < static_cast< long int >(n_in_[4]) &&
					       pos_x5 < static_cast< long int >(n_in_[5]) )
					    {
					      idx  = (x0 + w0) + static_cast< long int >(n_in_[0]) * (x1 + w1) + static_cast< long int >(n_in_[0]) * static_cast< long int >(n_in_[1]) * (x2 + w2);
					      idx +=  static_cast< long int >(n_in_[0]) * static_cast< long int >(n_in_[1]) * static_cast< long int >(n_in_[2]) * (x3 + w3);
					      idx +=  static_cast< long int >(n_in_[0]) * static_cast< long int >(n_in_[1]) * static_cast< long int >(n_in_[2]) * static_cast< long int >(n_in_[3]) * (x4 + w4);
					      idx +=  static_cast< long int >(n_in_[0]) * static_cast< long int >(n_in_[1]) * static_cast< long int >(n_in_[2]) * static_cast< long int >(n_in_[3]) *
						static_cast< long int >(n_in_[4]) * (x5 + w5);
					      weights_matrix_.insert( row, idx ) = widx++;
					    }
					  else
					    widx++;
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
	      std::string mess = "Building convolution windows with high dimensions has not been thought yet, but can be customized.\n";
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
  //
  //
  //
  template< class T, int D > void
  Window< T, D >::save_weights( std::ofstream& Weights_file  ) const
  {
    try
      {
	if ( Weights_file.is_open() )
	  {
	    //
	    // Save the information
	    Weights_file.write( (char*) (&k_), sizeof(int) );
	    Weights_file.write( (char*) (&number_weights_), sizeof(int) );
	    // Save the weights
	    for ( int k = 0 ; k < k_ ; k++ )
	      Weights_file.write( (char*)&weight_values_[k][0], number_weights_ * sizeof(T) );
	    //
	    // Save the position
	    typename Eigen::SparseMatrix< int, Eigen::RowMajor >::Index
	      rows = weights_matrix_.rows(),
	      cols = weights_matrix_.cols(),
	      nnzs = weights_matrix_.nonZeros(),
	      outS = weights_matrix_.outerSize(),
	      innS = weights_matrix_.innerSize();
	    //
	    Weights_file.write( reinterpret_cast<char*>( &rows), sizeof(typename Eigen::SparseMatrix< int, Eigen::RowMajor >::Index) );
	    Weights_file.write( reinterpret_cast<char*>( &cols), sizeof(typename Eigen::SparseMatrix< int, Eigen::RowMajor >::Index) );
	    Weights_file.write( reinterpret_cast<char*>( &nnzs), sizeof(typename Eigen::SparseMatrix< int, Eigen::RowMajor >::Index) );
	    Weights_file.write( reinterpret_cast<char*>( &outS), sizeof(typename Eigen::SparseMatrix< int, Eigen::RowMajor >::Index) );
	    Weights_file.write( reinterpret_cast<char*>( &innS), sizeof(typename Eigen::SparseMatrix< int, Eigen::RowMajor >::Index) );
	    //
	    typename Eigen::SparseMatrix< int, Eigen::RowMajor >::Index sizeIndexS = static_cast<typename Eigen::SparseMatrix< int, Eigen::RowMajor >::Index>(sizeof(typename Eigen::SparseMatrix< int, Eigen::RowMajor >::StorageIndex));
	    typename Eigen::SparseMatrix< int, Eigen::RowMajor >::Index sizeScalar = static_cast<typename Eigen::SparseMatrix< int, Eigen::RowMajor >::Index>(sizeof(typename Eigen::SparseMatrix< int, Eigen::RowMajor >::Scalar      ));
	    Weights_file.write( reinterpret_cast<const char* >( weights_matrix_.valuePtr()),      sizeScalar * nnzs );
	    Weights_file.write( reinterpret_cast<const char* >( weights_matrix_.outerIndexPtr()), sizeIndexS * outS );
	    Weights_file.write( reinterpret_cast<const char* >( weights_matrix_.innerIndexPtr()), sizeIndexS * nnzs );
	  }
	else
	    {
	      std::string mess = "The matrix file has no I/O access.\n";
	      throw MAC::MACException( __FILE__, __LINE__,
				       mess.c_str(),
				       ITK_LOCATION );
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
