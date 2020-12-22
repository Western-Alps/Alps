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
   * following the k-folds strategy for weigths adjustments.
   * 
   */
  template< typename Mountain, int Dim >
    class FullSamples : public Validation
    {
    public:
      /** Constructor. */
    explicit FullSamples();
      
      /**  */
      virtual ~FullSamples( )
	{};

      //
      // train the calssification engin
      virtual void train() override;
      // use the calssification engin
      virtual void use()   override;


    private:
      // Load the subjects
      Alps::Subjects< /*Functions,*/ Dim > subjects_{ std::make_shared< Mountain >() };
      // testing size
      int testing_size_{0};
      std::vector< std::list<int> > testing_set_;
      // training size
      int training_size_{0};
      std::vector< std::list<int> > training_set_;
    };

  //
  //
  template< typename M, int D >
  FullSamples< M, D >::FullSamples()
    {
      try
	{
//	  //
//	  // folds construction
//	  std::size_t
//	    number_of_subjects = Alps::LoadDataSet::instance()->get_data()["inputs"]["images"][0].size();
//	  // 
//	  if ( K > number_of_subjects )
//	    throw MAC::MACException( __FILE__, __LINE__,
//				     "It can't be more folds than subjects.",
//				     ITK_LOCATION );
//	  //
//	  testing_size_  = number_of_subjects / K;
//	  training_size_ = number_of_subjects - testing_size_;
//	  // resize the sets
//	  testing_set_.resize( K );
//	  training_set_.resize( K );
//	  //
//	  std::size_t offset = 0;
//	  for ( int kk = 0 ; kk < K ; kk++ )
//	    {
//	      for ( std::size_t s = 0 ; s < number_of_subjects ; s++ )
//		if ( s < testing_size_+offset && s >= offset )
//		  testing_set_[kk].push_back(s);
//		else
//		  training_set_[kk].push_back(s);
//	      //
//	      offset += testing_size_;
//	    }
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
//	  std::cout << "Multi-threading Cross Validation" << std::endl;
//	  // Start the pool of threads
//	  // Please do not remove the bracket!!
//	  {
//	    Alps::ThreadDispatching pool( K );
//	    for ( int k = 0 ; k < K ; k++ )
//	      pool.enqueue( std::ref( *this ), k );
//	  }
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
