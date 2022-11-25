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
#ifndef ALPSIMAGE_H
#define ALPSIMAGE_H
//
//
//
#include <iostream> 
#include <algorithm>
// ITK
#include "ITKHeaders.h"
//
#include "MACException.h"
#include "AlpsTensor.h"
#include "AlpsTools.h"
//
//
//
/*! \namespace Alps
 *
 * Name space Alps.
 *
 */
namespace Alps
{
  /*! \class Image
   *
   * \brief class Image record the information of the image through the processing. 
   * 
   * Images are flattened into a 1D tensor. Any dimension of image are going to 
   * be vectorized in an array of 1D.
   *
   */
  template< typename Type, int Dim >
  class Image : public Alps::Tensor< Type, 1 >
  {
    
  public:
    /** Constructor */
    Image() = default;
    /** Constructor */
    Image( const typename Reader< Dim >::Pointer );
    /** Constructor */
    Image( const std::vector< std::size_t >, const std::vector< double > );
    /** Constructor */
    Image( const std::array< std::size_t, Dim >, const std::vector< double > );
    /* Destructor */
    virtual ~Image() = default;


    
    //
    // Accessors
    //
    // From Alps::Tensor< Type, 1 >
    //
    //! Get size of the tensor
    virtual const std::vector< std::size_t > get_tensor_size() const noexcept                 override
    {
      if ( tensor_size_[0] != tensor_.size() )
	{
	  std::cout
	    << "tensor_size_[0] " << tensor_size_[0]
	    << " tensor_.size() " << tensor_.size()
	    << std::endl;
	  throw MAC::MACException( __FILE__, __LINE__,
				   "The tensor size changed for this layer.",
				   ITK_LOCATION );
	}
      //
      //
      return tensor_size_;
    };
    //! Get the tensor
    virtual const std::vector< Type >&       get_tensor() const noexcept                      override 
    { return tensor_;};
    //! Update the tensor
    virtual std::vector< Type >&             update_tensor()                                  override 
    { return tensor_;};
    //
    // From Image< typename Type, int Dim >
    //
    //! Get region from the original image
    /*!
      \return region descriptor from ITK
    */
    virtual const typename ImageType< Dim >::RegionType get_image_region() const noexcept
    { return region_;};
    //! Get start from the original image
    /*!
      \return the starting point from ITK region descriptor
    */
    virtual const typename ImageType< Dim >::IndexType  get_image_start() const noexcept
    { return start_;};
    //! Get size of the original image
    /*!
      \return the image size from ITK region descriptor
    */
    virtual const typename ImageType< Dim >::SizeType   get_image_size() const noexcept
    { return size_;};


    
    //
    // Functions
    //
    // From Alps::Tensor< Type, 1 >
    //
    //! Save the tensor values (e.g. weights)
    virtual void save_tensor( std::ofstream& ) const{};
    //! Load the tensor values (e.g. weights)
    virtual void load_tensor( const std::ofstream& ) {};
    //
    // From Image< typename Type, int Dim >
    //
    //! Implementation of [] operator.  This function must return a 
    // reference as array element can be put on left side 
    //! Implementation of [] operator
    /*!
      The member returns the array element of the vectorized image.
      \param Index of the element 
      \return the iamge value
    */
    Type         operator[]( const std::size_t Idx );
    // Visualization of the image
    void         visualization( const std::string,
				const typename Reader< Dim >::Pointer );




    
  private:
    //
    // Image properties
    //
    // Image region
    typename ImageType< Dim >::RegionType region_;
    // Starting point
    typename ImageType< Dim >::IndexType  start_;
    // Size in each dimension
    typename ImageType< Dim >::SizeType   size_;

    
    //
    // Neural network properties
    //
    std::vector< std::size_t > tensor_size_{ std::vector< std::size_t >(/*tensor order*/1,1) };
    // Z
    std::vector< Type >        tensor_;

    //
    // Output image
    ImageType< Dim >::Pointer output_image_;
  };
  //
  //
  // Constructor
  template< typename T,int D >
  Alps::Image< T, D >::Image( const typename Reader< D >::Pointer Image_reader )
  {
    try
      {
	//
	// Normalization indexes
	// We want the input images between [0,1]
	typename ImageCalculatorFilterType< D >::Pointer imageCalculatorFilter = ImageCalculatorFilterType< D >::New();
	imageCalculatorFilter->SetImage( Image_reader->GetOutput() );
	imageCalculatorFilter->Compute();
	//
	typename ImageType< D >::IndexType minLocation = imageCalculatorFilter->GetIndexOfMinimum();
	typename ImageType< D >::IndexType maxLocation = imageCalculatorFilter->GetIndexOfMaximum();
	//
	double
	  max = static_cast<double>( Image_reader->GetOutput()->GetPixel(maxLocation) ),
	  min = static_cast<double>( Image_reader->GetOutput()->GetPixel(minLocation) );
	double delta = max - min;
	// Check the image is not empty
	if ( delta == 0. )
	  throw MAC::MACException( __FILE__, __LINE__,
				   "The image holds the same value all over.",
				   ITK_LOCATION );


 
	//
	// Create the region
	//
	size_ = Image_reader->GetOutput()->GetLargestPossibleRegion().GetSize();
	for ( int d = 0 ; d < D ; d++ )
	  {
	    start_[d]        = 0;
	    tensor_size_[0] *= size_[d];
	  }
	//
	// Resize elements
	region_.SetSize( size_ );
	region_.SetIndex( start_ );
	//
	tensor_ = std::vector< T >( tensor_size_[0], 0. );
	//
	// Create the output image
	typename DuplicatorType< D >::Pointer input_to_output = DuplicatorType< D >::New();
	input_to_output->SetInputImage( Image_reader->GetOutput() );
	input_to_output->Update();
	//
	output_image_ = input_to_output->GetOutput();
	
	//
	//
	ImageRegionIterator< ImageType< D > > imageIterator( Image_reader->GetOutput(),
							     region_ );
	std::size_t position = 0;
	while( !imageIterator.IsAtEnd() )
	  {
	    tensor_[ position++ ] = (imageIterator.Value() - min) / delta;
	    ++imageIterator;
	  }
	// Check the vector has been created correctly
	if ( position != tensor_size_[0] )
	  throw MAC::MACException( __FILE__, __LINE__,
				   "The image tensor has not been created correctly.",
				   ITK_LOCATION );
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
      }
  }
  //
  //
  // Constructor
  template< typename T,int D >
  Alps::Image< T, D >::Image( const std::vector< std::size_t > Tensor_size,
			      const std::vector< double >      Tensor ):
    tensor_size_{Tensor_size}, tensor_{Tensor}
  {}
  //
  //
  // Constructor
  template< typename T,int D >
  Alps::Image< T, D >::Image( const std::array< std::size_t, D > Tensor_size,
			      const std::vector< double >        Tensor ):
    tensor_{Tensor}
  {
    //
    // Create the region
    //
    for ( int d = 0 ; d < D ; d++ )
      {
	size_[d]         = Tensor_size[d];
	start_[d]        = 0;
	tensor_size_[0] *= size_[d];
      }
    //
    // Resize elements
    region_.SetSize( size_ );
    region_.SetIndex( start_ );
  }
  //
  //
  // Operator []
  template< typename T,int D > T
  Alps::Image< T, D >::operator[]( const std::size_t Idx )
  {
    try
      {
	if ( Idx < 0 || Idx > tensor_size_[0] )
	  throw MAC::MACException( __FILE__, __LINE__,
				   "Indexing out of bound.",
				   ITK_LOCATION );
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
      }
    //
    //
    return tensor_[Idx]; 
  }
  //
  //
  // 
  template< typename T,int D > void
  Alps::Image< T, D >::visualization( const std::string Name,
				       const typename Reader< D >::Pointer Image_reader )
  {
   //using ImageType = itk::Image< T, D >;
   using WriterType = itk::ImageFileWriter< ImageType<D> >;

   //
   // Create the output image
   typename DuplicatorType< D >::Pointer input_to_output = DuplicatorType< D >::New();
   input_to_output->SetInputImage( Image_reader->GetOutput() );
   input_to_output->Update();
   //
   output_image_ = input_to_output->GetOutput();
   //
   size_ = Image_reader->GetOutput()->GetLargestPossibleRegion().GetSize();
   for ( int d = 0 ; d < D ; d++ )
     start_[d]        = 0;
   // Resize elements
   region_.SetSize( size_ );
   region_.SetIndex( start_ );

   //
   //
    ImageRegionIterator< ImageType<D> > imageIterator( output_image_, region_ );
    std::size_t position = 0;
    while( !imageIterator.IsAtEnd() )
      {
	imageIterator.Set( tensor_[ position++ ] );
	++imageIterator;
      }
    // Check the vector has been created correctly
    if ( position != tensor_size_[0] )
      throw MAC::MACException( __FILE__, __LINE__,
			       "The image tensor has not been created correctly.",
			       ITK_LOCATION );
   
   //
   //
   itk::NiftiImageIO::Pointer   nifti_io_cortex = itk::NiftiImageIO::New();
   typename WriterType::Pointer writer = WriterType::New();
   writer->SetFileName( Name );
   writer->SetInput( output_image_ );
   writer->SetImageIO( nifti_io_cortex );
   //
   try
     {
	writer->Update();
     }
   catch( itk::ExceptionObject & err )
     {
	std::cerr << err << std::endl;
     }
  }
}
#endif
