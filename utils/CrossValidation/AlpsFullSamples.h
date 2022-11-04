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
#ifndef APLSFULL_H
#define APLSFULL_H
//
//
//
#include <iostream>
#include <fstream>
#include <list>
//
//
//
#include "MACException.h"
#include "AlpsSubjects.h"
#include "AlpsValidation.h"
#include "AlpsLoadDataSet.h"
//
//
//
namespace Alps
{
  /** \class FullSamples
   *
   * \brief FullSamples class is one of the validation class
   * following the full samples strategy for weigths adjustments.
   * The template arguments cover:
   * - Mountain: ont of the predefined neural network architecture
   * - Dim: Dimesion of the images
   *
   */
  template< typename Mountain, int Dim >
    class FullSamples : public Validation
    {
    public:
      /** Constructor. */
    explicit FullSamples();
      
      /**  Destructor. */
      virtual ~FullSamples() = default;

      
      //
      // Functions
      //
      // train the calssification engin
      virtual void train() override;
      // use the calssification engin
      virtual void use()   override;


    private:
      // Load the subjects
      std::vector< Alps::Subjects< /*Functions,*/ Dim > > subjects_; 
      // testing size
      int testing_size_{0};
      std::vector< std::list<int> >                       testing_set_;
      // training size
      int training_size_{0};
      std::vector< std::list<int> >                       training_set_;
      // training frequency to dump the reusults
      std::size_t training_dump_{0};
    };

  //
  //
  template< typename M, int D >
  FullSamples< M, D >::FullSamples()
    {
      try
	{
	  //
	  // folds construction
	  std::size_t
	    number_of_subjects = Alps::LoadDataSet::instance()->get_data()["inputs"]["images"][0].size();
	  //
	  training_dump_     = Alps::LoadDataSet::instance()->get_data()["mountain"]["strategy"]["dump_results"];
	  //
	  testing_size_  = 0;
	  training_size_ = number_of_subjects;
	  // resize the sets
	  training_set_.resize( 1 );
	  //
	  for ( std::size_t s = 0 ; s < number_of_subjects ; s++ )
	    training_set_[0].push_back(s);
	  // resize the subjects vector
	  subjects_.push_back( Alps::Subjects< /*Functions,*/ D >(std::make_shared< M >()) );
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	}
    }
  //
  //
  template< typename M, int D > void
  FullSamples< M, D >::train()
    {
      try
	{
	  //
	  //
	  std::list< int > fold_subjects =  training_set_[0];
	  double
	    energy_previous_epoque = 1.e+06;
	  while ( std::dynamic_pointer_cast<M>(subjects_[0].get_mountain())->get_energy() > 1.e-06 )
	    {
	      //
	      //
	      double relat = 100. * std::fabs(energy_previous_epoque - std::dynamic_pointer_cast<M>(subjects_[0].get_mountain())->get_energy()) / energy_previous_epoque;
	      //
	      std::cout
		<< "Epoque " << subjects_[0].get_epoque() << " cost function: "
		<< std::dynamic_pointer_cast<M>(subjects_[0].get_mountain())->get_energy()
		<< " Relative difference in cost function: "
		<< relat
		<< std::endl;
	      // Reset the current epoque to the previous epoque
	      energy_previous_epoque = std::dynamic_pointer_cast<M>(subjects_[0].get_mountain())->get_energy();
		
	      /////////////////////
	      // Forward process //
	      /////////////////////
	      //
	      // The end of the forward process, each image generate a cost (error for each image)
	      // The notify does the work for that.
	      for ( int sub : fold_subjects )
		{
		  std::cout
		    << " Subject " << sub
		    << " -- fwd: In subject " << ( subjects_[0].get_subjects() )[ sub ]->get_subject_number()
		    << std::endl;
		  // Get the observed Mountain from subjects
		  // Then foward across the neural network. The last layer estimate the cost function
		  std::dynamic_pointer_cast<M>( subjects_[0].get_mountain() )->forward( (subjects_[0].get_subjects())[sub] );
		}

	      
	      //////////////////////
	      // Backward process //
	      //////////////////////
	      //
//	      for ( int sub : fold_subjects )
//		{
		  std::cout
		    << " Subject " << 0
		    << " -- bwd: In subject " << ( subjects_[0].get_subjects() )[ 0 ]->get_subject_number()
		    << std::endl;
		  // Get the observed Mountain from subjects
		  // Backward process
		  std::dynamic_pointer_cast<M>( subjects_[0].get_mountain() )->backward( (subjects_[0].get_subjects())[0] );
//		}
//
//	      //
//	      // The updates of the weights can be done at the end of a batch of images.
//	      // In case the size of the batch and the epoque does not coincide, we force the update.
//	      for ( int sub : fold_subjects )
//		{
//		  std::cout
//		    << " Subject " << sub
//		    << " -- Weight update In subject " << ( subjects_[0].get_subjects() )[ sub ]->get_subject_number()
//		    << std::endl;
//		  // Get the observed Mountain from subjects
//		  // Update the weights
//		  std::dynamic_pointer_cast<M>( subjects_[0].get_mountain() )->weight_update( (subjects_[0].get_subjects())[sub] );
//		}


	      //
	      // update the epoques
	      // By updating the epoque the calculation of the cost function 
	      ++subjects_[0];
	      if ( subjects_[0].get_epoque() % training_dump_ == 0 )
		subjects_[0].visualization();
	    }
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	}
    }

  //
  //
  template< typename M, int D > void
  FullSamples< M, D >::use()
  {}
}
#endif
