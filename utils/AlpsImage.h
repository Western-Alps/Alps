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
  template< /*class Function,*/ typename Type, int Dim >
  class Image : public Alps::Tensor< Type, 1 >
  {
  public:
    /** Constructor */
    Image(){};
    /** Constructor */
    Image( const typename Reader< Dim >::Pointer );
    /* Destructor */
    virtual ~Image(){};

    
    //
    // Accessors
    //
    // Get size of the tensor
    virtual const std::vector< std::size_t > get_tensor_size()                              const
      { return array_size_;};
    // Get the tensor
    virtual std::shared_ptr< Type >          get_tensor()                                   const
    { return z_;};
    // Set size of the tensor
    virtual void                             set_tensor_size( std::vector< std::size_t > S)
    { array_size_ = S;};
    // Set the tensor
    virtual void                             set_tensor( std::shared_ptr< Type > Z )
    { z_ = Z;};


    //
    // Functions
    //
    // Save the tensor values (e.g. weights)
    virtual void save_tensor() const{};
    // Load the tensor values (e.g. weights)
    virtual void load_tensor()      {};

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
    std::vector< std::size_t > array_size_{ std::vector< std::size_t >(1,1) };
    // Z
    std::shared_ptr< Type >    z_;
  };
  //
  //
  // Constructor
  template< /*class F,*/ typename T,int D >
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
	    array_size_[0] *= size_[d];
	  }
	//
	// Resize elements
	region_.SetSize( size_ );
	region_.SetIndex( start_ );
	//
	z_   = std::shared_ptr< T >( new  T [array_size_[0]], std::default_delete<  T [] >() );
	//
	ImageRegionIterator< ImageType< D > > imageIterator( Image_reader->GetOutput(),
							     region_ );
	int position = 0;
	while( !imageIterator.IsAtEnd() )
	  {
	    ( z_.get() )[ position++ ] = imageIterator.Value();
	    ++imageIterator;
	  }
	// Check the vector has been created correctly
	if ( position != array_size_[0] )
	  throw MAC::MACException( __FILE__, __LINE__,
				   "The iamge vector has not been created correctly.",
				   ITK_LOCATION );
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
      }
  }
}
#endif
