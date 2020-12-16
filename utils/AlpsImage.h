#ifndef ALPSIMAGE_H
#define ALPSIMAGE_H
//
//
//
#include <iostream> 
//
// 
//
#include "MACException.h"
#include "AlpsMountain.h"
#include "AlpsClimber.h"
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

    //
    /** Constructor */
    Image( std::shared_ptr< Alps::Mountain >,
	   const std::string ){};
    /** Constructor */
    Image( std::shared_ptr< Alps::Mountain >,
	   const std::vector< std::size_t > );
    /* Destructor */
    virtual ~Image(){};

    
    //
    // Accessors

    //
    // Overrided function
    // Update the subject information
    virtual void update() override {};


  private:
    // Dimension of the image along all directions
    std::vector< std::size_t > size_;
    // Attached mountain
    std::shared_ptr< Alps::Mountain > mountain_observed_{nullptr};
  };
  
  //
  // Constructor
  template< /*class F,*/ int Dim >
  Alps::Image<Dim>::Image( std::shared_ptr< Alps::Mountain > Mountain,
			   const std::vector< std::size_t >  Size ):
    size_{Size}, mountain_observed_{Mountain}
  {
    try
      {
	// Check the dimensions are fine
	if ( Size.size() == Dim )
	  {/* Do Something */}
	else
	  throw MAC::MACException( __FILE__, __LINE__,
				     "The dimensions are different.",
				     ITK_LOCATION );
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
      }
  }
}
#endif
