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
   * - MiniBatch: the size of the mini-bach
   *   * MiniBatch =  1 -- stochastic gradient descent
   *   * MiniBatch =  n -- batch of size n images, n < N the total 
   *                       number of images
   *   * MiniBatch = -1 -- the model uses all the images
   * - Dim: Dimesion of the images
   *
   */
  template< typename Mountain, int MiniBatch, int Dim >
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
  template< typename M, int MnB, int D >
  FullSamples< M, MnB, D >::FullSamples()
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
  template< typename M, int MnB, int D > void
  FullSamples< M, MnB, D >::train()
    {
      try
	{
	  //
	  //
	  std::list< int > fold_subjects =  training_set_[0];
	  if ( MnB > training_size_ + testing_size_ )
	    throw MAC::MACException( __FILE__, __LINE__,
				     "The size of a mini-batch can't exceed the number or images. Option -1 takes the entire dataset.",
				     ITK_LOCATION );
	  // All the dataset
	  if ( MnB == -1 )
	    {
	      while ( std::dynamic_pointer_cast<M>(subjects_[0].get_mountain())->get_energy() > 1.e-06 )
		{
		  //
		  // Forward process
		  for ( int sub : fold_subjects )
		    {
		      std::cout
			<< " Subject " << sub
			<< " -- In subject " << ( subjects_[0].get_subjects() )[ sub ]->get_subject_number()
			<< std::endl;
		      // Get the observed Mountain from subjects
		      std::dynamic_pointer_cast<M>( subjects_[0].get_mountain() )->forward( (subjects_[0].get_subjects())[sub] );
		    }
		  // update the epoques
		  training_set_[0]++;

		  //
		  // Estimate the cost function
		  
		  
		  //
		  // Backward process
		}
	    }
	  // minibatch size n
	  else
	    {
	      //
	      // ToDo make a cutoff on after n images to update the weights
	      while ( std::dynamic_pointer_cast<M>(subjects_[0].get_mountain())->get_energy() > 1.e-06 )
		for ( int sub : fold_subjects )
		  {
		    std::cout
		      << " Subject " << sub
		      << " -- In subject " << ( subjects_[0].get_subjects() )[ sub ]->get_subject_number()
		      << std::endl;
		    // Get the observed Mountain from subjects
		    std::dynamic_pointer_cast<M>( subjects_[0].get_mountain() )->forward( (subjects_[0].get_subjects())[sub] );
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
  template< typename M, int MnB, int D > void
  FullSamples< M, MnB, D >::use()
  {}
}
#endif
