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
      virtual ~FullSamples(){};

      
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
	  while ( std::dynamic_pointer_cast<M>(subjects_[0].get_mountain())->get_energy() > 1.e-06 )
	    {
	      std::cout
		<< "Epoque cost function: "
		<< std::dynamic_pointer_cast<M>(subjects_[0].get_mountain())->get_energy()
		<< std::endl;
		
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
	      for ( int sub : fold_subjects )
		{
		  std::cout
		    << " Subject " << sub
		    << " -- bwd: In subject " << ( subjects_[0].get_subjects() )[ sub ]->get_subject_number()
		    << std::endl;
		  // Get the observed Mountain from subjects
		  // Backward process
		  std::dynamic_pointer_cast<M>( subjects_[0].get_mountain() )->backward( (subjects_[0].get_subjects())[sub] );
		}

	      //
	      // The updates of the weights can be done at the end of a batch of images.
	      // In case the size of the batch and the epoque does not coincide, we force the update.
	      for ( int sub : fold_subjects )
		{
		  std::cout
		    << " Subject " << sub
		    << " -- Weight update In subject " << ( subjects_[0].get_subjects() )[ sub ]->get_subject_number()
		    << std::endl;
		  // Get the observed Mountain from subjects
		  // Backward process
		  std::dynamic_pointer_cast<M>( subjects_[0].get_mountain() )->weight_update( (subjects_[0].get_subjects())[sub] );
		}


	      //
	      // update the epoques
	      // By updating the epoque the calculation of the cost function 
	      ++subjects_[0];
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
