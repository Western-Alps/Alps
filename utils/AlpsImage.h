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
#include "AlpsClimber.h"
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
  class Image : public Alps::Climber
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
    //
    // Get the observed mountain
    virtual std::shared_ptr< Alps::Mountain >                get_mountain()                            override
    { return nullptr;};
    // Get layer modality
    virtual std::vector< std::shared_ptr< Alps::Climber > >& get_layer_modalities( const std::string ) override
    {};
    //
    // Get the size of the array
    const int              get_array_size()                   const
    { return array_size_;};
    // Get Z
    std::shared_ptr< double > get_z()                         const 
    { return z_;};
    // Get the error vector
    std::shared_ptr< double > get_eps()                       const
    { return eps_;};
    // Get Z
    void                   set_z( std::shared_ptr< double > );
    void                   set_z( std::vector< double > );
    // Get the error vector
    void                  set_eps( std::shared_ptr< double > );
    void                  set_eps( std::vector< double > );


    //
    // Functions
    // Update the subject information
    virtual void                                             update()                                  override{};

    
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
    int                       array_size_{1};
    // Z
    std::shared_ptr< double > z_;
    // error
    std::shared_ptr< double > eps_;
  };
  //
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
	for ( int d = 0 ; d < D ; d++ )
	  {
	    start_[d]    = 0;
	    array_size_ *= size_[d];
	  }
	//
	// Resize elements
	region_.SetSize( size_ );
	region_.SetIndex( start_ );
	//
	z_   = std::shared_ptr<double>( new double[array_size_], std::default_delete< double[] >() );
	eps_ = std::shared_ptr<double>( new double[array_size_], std::default_delete< double[] >() );
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
	if ( position != array_size_ )
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
  // 
  template< /*class F,*/ int D > void
  Alps::Image< D >::set_z( std::shared_ptr< double > Z )
  {
    z_ = Z;
  }
  //
  //
  // 
  template< /*class F,*/ int D > void
  Alps::Image< D >::set_z( std::vector< double > Z )
  {
    //
    //
    std::size_t s = Z.size();
    z_ = std::shared_ptr<double>( new double[s], std::default_delete< double[] >() );
    for ( std::size_t i = 0 ; i < s ; i++ )
      ( z_.get() )[i] = Z[i];
  }
  //
  //
  // 
  template< /*class F,*/ int D > void
  Alps::Image< D >::set_eps( std::shared_ptr< double > Eps )
  {
    eps_ = Eps;
  }
  //
  //
  // 
  template< /*class F,*/ int D > void
  Alps::Image< D >::set_eps( std::vector< double > Eps )
  {
    std::size_t s = Eps.size();
    eps_ = std::shared_ptr<double>( new double[s], std::default_delete< double[] >() );
    for ( std::size_t i = 0 ; i < s ; i++ )
      ( eps_.get() )[i] = Eps[i];
  }
}
#endif
