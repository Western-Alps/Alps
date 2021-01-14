#ifndef ALPSSUBJECTS_H
#define ALPSSUBJECTS_H
//
//
//
#include <iostream> 
#include <memory>
//
// 
//
#include "MACException.h"
#include "AlpsLoadDataSet.h"
#include "AlpsClimber.h"
#include "AlpsMountain.h"
#include "AlpsSubject.h"
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
  /*! \class Subjects
   *
   * \brief class Subjects (observer) is a mountain observer and inherite 
   * from Climber behavioral observation pattern.
   *
   */
  // Observer
  template< /*class Function,*/ int Dim >
  class Subjects : public Alps::Climber
  {
  public:

    //
    /** Constructor */
    explicit Subjects( std::shared_ptr< Alps::Mountain > );
    /* Destructor */
    virtual ~Subjects(){};

    
    //
    // Accessors
    //
    // Get the observed mountain
    virtual std::shared_ptr< Alps::Mountain >                      get_mountain()       override
    { return mountain_observed_;};
    //
    //
    const   std::vector< std::shared_ptr< Alps::Subject< Dim > > > get_subjects() const 
    { return subjects_;};

    
    //
    // functions
    //
    virtual void                                                   update()             override;
    //
    //
    // prefix increment
    Subjects& operator++()
    {
      epoque_++;
      return *this; 
    }

    
  private:
    // Number of epoques
    std::size_t                                            epoque_{0};
    // Attached observed Mountain
    std::shared_ptr< Alps::Mountain >                      mountain_observed_;
    // 
    std::vector< std::shared_ptr< Alps::Subject< Dim > > > subjects_;
    // This function is the continuous step function
    /*Function activation_function_;*/
  };

  //
  // Constructor
  template< /*class F,*/ int D >
  Alps::Subjects</*A,*/ D >::Subjects( std::shared_ptr< Alps::Mountain > Mountain ):
    mountain_observed_{Mountain}
  {
    try
      {
	//
	// Create subject and load the images
	int
	  subject_number  = 0;
	std::size_t
	  num_modalities  = Alps::LoadDataSet::instance()->get_data()["inputs"]["images"].size(),
	  num_img_per_mod = Alps::LoadDataSet::instance()->get_data()["inputs"]["images"][0].size();
	//
	//
	for ( std::size_t img = 0 ; img < num_img_per_mod ; img++ )
	  {
	    // create the subject
	    subjects_.push_back( std::make_shared< Alps::Subject< D > >(subject_number++, num_modalities) );
	    // record the modality for each subject
	    for ( std::size_t mod = 0 ; mod < num_modalities ; mod++ )
	      ( subjects_[subject_number-1].get() )->add_modalities( Alps::LoadDataSet::instance()->get_data()["inputs"]["images"][mod][img] );
	    // Check everything is fine
	    if ( !subjects_[subject_number-1].get()->check_modalities("__input_layer__") )
	      {
		std::string mess = "The number of modalities expected is different from the number of loaded modalities.";
		throw MAC::MACException( __FILE__, __LINE__,
					 mess.c_str(),
					 ITK_LOCATION );
	      }
	  }
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
  Alps::Subjects< D >::update()
  {
    try
      {
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
      }
  }
}
#endif
