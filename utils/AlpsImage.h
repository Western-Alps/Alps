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
   * \brief class Image record the information 
   * of the image through the processing. Images are
   * tensor 1D. Any dimension of image are going to 
   * be vectorized in an array of 1D.
   *
   */
  template< typename Type, int Dim >
  class Image : public Alps::Tensor< Type, 1 >
  {
  public:
    /** Constructor */
    Image(){};
    /** Constructor */
    Image( const typename Reader< Dim >::Pointer );
    /** Constructor */
    Image( const std::vector< std::size_t >, std::shared_ptr< double > );
    /* Destructor */
    virtual ~Image(){};

    
    //
    // Accessors
    //
    // Get size of the tensor
    virtual const std::vector< std::size_t > get_tensor_size() const                        override
      { return tensor_size_;};
    // Get the tensor
    virtual std::shared_ptr< Type >          get_tensor() const                             override 
    { return tensor_;};
    // Set size of the tensor
    virtual void                             set_tensor_size( std::vector< std::size_t > S) override
    { tensor_size_ = S;};
    // Set the tensor
    virtual void                             set_tensor( std::shared_ptr< Type > Z )        override
    { tensor_ = Z;};


    //
    // Functions
    //
    // Save the tensor values (e.g. weights)
    virtual void save_tensor() const{};
    // Load the tensor values (e.g. weights)
    virtual void load_tensor( const std::string ) {};

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
    std::shared_ptr< Type >    tensor_{nullptr};
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
	// Create the region
	//
	size_ = Image_reader->GetOutput()->GetLargestPossibleRegion().GetSize();
	for ( int d = 0 ; d < D ; d++ )
	  {
	    start_[d]       = 0;
	    tensor_size_[0] *= size_[d];
	  }
	//
	// Resize elements
	region_.SetSize( size_ );
	region_.SetIndex( start_ );
	//
	tensor_ = std::shared_ptr< T >( new  T [tensor_size_[0]], std::default_delete<  T [] >() );
	//
	ImageRegionIterator< ImageType< D > > imageIterator( Image_reader->GetOutput(),
							     region_ );
	std::size_t position = 0;
	while( !imageIterator.IsAtEnd() )
	  {
	    ( tensor_.get() )[ position++ ] = imageIterator.Value();
	    ++imageIterator;
	  }
	// Check the vector has been created correctly
	if ( position != tensor_size_[0] )
	  throw MAC::MACException( __FILE__, __LINE__,
				   "The iamge vector has not been created correctly.",
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
			      std::shared_ptr< double >        Tensor ):
    tensor_size_{Tensor_size}, tensor_{Tensor}
  {}
}
#endif
