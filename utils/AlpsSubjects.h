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
#include "AlpsClimber.h"
#include "AlpsMountain.h"
#include "AlpsSubject.h"
#include "AlpsSubjectCPU.h"
#include "AlpsSubjectGPU.h"
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
  template< /*class Function,*/ int Arch, int Dim >
  class Subjects : Alps::Climber
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
    // functions
    //
    virtual void update( std::shared_ptr< Alps::Mountain > );

  private:
    // Attached observed Mountain
    std::shared_ptr< Alps::Mountain > mountain_observed_;
    
    // 
    std::vector< std::shared_ptr< Alps::Subject > > subjects_;
    // This function is the continuous step function
    /*Function activation_function_;*/
  };

  //
  // Constructor
  template< /*class F,*/ int A, int Dim >
  Alps::Subjects<A,Dim>::Subjects( std::shared_ptr< Alps::Mountain > Mountain):
    Alps::Climber(),
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
	switch ( A )
	  {
	  case Alps::Architecture::GPU:
	    {
	      break;
	    }
	  case Alps::Architecture::CPU:
	    {
	      //
	      for ( std::size_t img = 0 ; img < num_img_per_mod ; img++ )
		{
		  std::cout << std::endl;
		  // create the subject
		  subjects_.push_back( std::make_shared< Alps::SubjectCPU< Dim > >(subject_number++, num_modalities) );
		  // record the modality for each subject
		  for ( std::size_t mod = 0 ; mod < num_modalities ; mod++ )
		    ( subjects_[subject_number-1].get() )->add_modalities( Alps::LoadDataSet::instance()->get_data()["inputs"]["images"][mod][img] );
		  // Check everything is fine
		  if ( !subjects_[subject_number-1].get()->check_modalities() )
		    {
		      std::string mess = "The number of modalities expected is different from the number of loaded modalities.";
		      throw MAC::MACException( __FILE__, __LINE__,
					       mess.c_str(),
					       ITK_LOCATION );
		    }
		}
	      //
	      break;
	    }
	  default:
	    {
	      std::string mess = "The architecture (CPU/GPU) must be specified.";
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
      }
  }
  //
  // 
  template< /*class F,*/ int A, int Dim > void
  Alps::Subjects<A,Dim>::update( std::shared_ptr< Alps::Mountain > Mount )
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
