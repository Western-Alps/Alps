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
  template< int Dim >
  class Subjects : public Alps::Climber
  {
  public:

    //
    /** Constructor */
    explicit Subjects( std::shared_ptr< Alps::Mountain > );
    /* Destructor */
    virtual ~Subjects() = default;

    
    //
    // Accessors
    //
    // Get the observed mountain
    virtual std::shared_ptr< Alps::Mountain >                      get_mountain()       override
    { return mountain_observed_;};
    // get images energy
    virtual const double                                           get_energy() const   override
    { return /*ToDo*/-1.;};
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
    Subjects&                                                      operator++();

    
  private:
    // Number of epoques
    std::size_t                                            epoque_{0};
    // Attached observed Mountain
    std::shared_ptr< Alps::Mountain >                      mountain_observed_;
    // 
    std::vector< std::shared_ptr< Alps::Subject< Dim > > > subjects_;
  };

  //
  // Constructor
  template< int D >
  Alps::Subjects< D >::Subjects( std::shared_ptr< Alps::Mountain > Mountain ):
    mountain_observed_{Mountain}
  {
    try
      {
	//
	// Attache the observer in the observed mountain
	//mountain_observed_->attach( std::shared_ptr< Alps::Subjects< D > >(this) );
	//
	// Create subject and load the images
	int
	  subject_number  = 0;
	std::size_t
	  num_modalities  = Alps::LoadDataSet::instance()->get_data()["inputs"]["images"].size(),
	  num_img_per_mod = Alps::LoadDataSet::instance()->get_data()["inputs"]["images"][0].size(),
	  num_trg_per_mod = Alps::LoadDataSet::instance()->get_data()["inputs"]["image_targets"][0].size(),
	  label_universe  = static_cast< std::size_t >(Alps::LoadDataSet::instance()->get_data()["inputs"]["labels_universe"]);

	//
	//
	for ( std::size_t img = 0 ; img < num_img_per_mod ; img++ )
	  {
	    // create the subject
	    subjects_.push_back( std::make_shared< Alps::Subject< D > >(subject_number++, num_modalities) );
	    //
	    // If the output is descrete loop over the labels
	    if ( label_universe > 0 )
	      {
		std::string subject_label = "label_subject_" + std::to_string(img);
		std::size_t label = static_cast< std::size_t >( Alps::LoadDataSet::instance()->get_data()["inputs"]["labels"][subject_label] );
		( subjects_[subject_number-1].get() )->add_target( label, label_universe );
	      }
	    // record the target images
	    if ( num_trg_per_mod != 0)
	      {
		if( num_trg_per_mod == num_img_per_mod )
		  for ( std::size_t mod = 0 ; mod < num_modalities ; mod++ )
		    ( subjects_[subject_number-1].get() )->add_target( Alps::LoadDataSet::instance()->get_data()["inputs"]["image_targets"][mod][img] );
		else
		  {
		    std::string mess = "The number of targets must be the same as the number of images.";
		    throw MAC::MACException( __FILE__, __LINE__,
					     mess.c_str(),
					     ITK_LOCATION );
		  }
	      }
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
    mountain_observed_->notify();
  }
  //
  //
  //
  template< /*class F,*/ int D > Alps::Subjects< D >&
  Alps::Subjects< D >::operator++()
  {
    //
    // increment the epoque, then update
    epoque_++;
    update();

    //
    //
    return *this; 
  }
}
#endif
