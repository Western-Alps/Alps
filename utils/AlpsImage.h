#ifndef ALPSIMAGE_H
#define ALPSIMAGE_H
//
//
//
#include <iostream> 
// ITK
#include "ITKHeaders.h"
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
//
#include "MACException.h"
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
   * of the image through the processing.
   *
   */
  template< /*class Function,*/ int Dim >
  class Image 
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
    // Get Z
    std::shared_ptr< Eigen::MatrixXd > get_z()   const
    { return z_;};
    // Get the error vector
    std::shared_ptr< Eigen::MatrixXd > get_eps() const
    { return eps_;};
    // Get Z
    void set_z( std::shared_ptr< Eigen::MatrixXd > Z )
    { z_ = Z;};
    // Get the error vector
    void set_eps( std::shared_ptr< Eigen::MatrixXd > Epsilon )
    { eps_ = Epsilon;};


    //
    // Functions
    //

    
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
    // Z
    std::shared_ptr< Eigen::MatrixXd > z_;
    // error
    std::shared_ptr< Eigen::MatrixXd > eps_;
  };
  
  //
  // Constructor
  template< /*class F,*/ int D >
  Alps::Image< D >::Image( const typename Reader< D >::Pointer Image_reader )
  {
    try
      {
	//
	// Create the region
	//
	size_ = Image_reader->GetOutput()->GetLargestPossibleRegion().GetSize();
	int array_size = 1;
	for ( int d = 0 ; d < D ; d++ )
	  {
	    start_[d]   = 0;
	    array_size *= size_[d];
	  }
	//
	// Resize elements
	region_.SetSize( size_ );
	region_.SetIndex( start_ );
	//
	z_   = std::make_shared<Eigen::MatrixXd>( Eigen::MatrixXd::Zero(array_size,1) );
	eps_ = std::make_shared<Eigen::MatrixXd>( Eigen::MatrixXd::Zero(array_size,1) );
	//
	ImageRegionIterator< ImageType< D > > imageIterator( Image_reader->GetOutput(),
							     region_ );
	int position = 0;
	while( !imageIterator.IsAtEnd() )
	  {
	    (*z_)( position++, 0 ) = imageIterator.Value();
	    ++imageIterator;
	  }
	// Check the vector has been created correctly
	if ( position != array_size )
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
