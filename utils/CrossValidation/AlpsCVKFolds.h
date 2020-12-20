#ifndef APLSCVKFOLDS_H
#define APLSCVKFOLDS_H
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
#include "AlpsValidation.h"
#include "AlpsLoadDataSet.h"
//
//
//
namespace Alps
{
  /** \class CVKFolds
   *
   * \brief CVKFolds class is one of the validation class
   * following the k-folds strategy for weigths adjustments.
   * 
   */
  template< int K, typename Mountain >
    class CVKFolds : public Validation
    {
    public:
      /** Constructor. */
    explicit CVKFolds();
      
      /**  */
      virtual ~CVKFolds( )
	{};

      //
      // train the calssification engin
      virtual void train() override;
      // use the calssification engin
      virtual void use()   override;

    private:
//      //
//      // Solver used in the cross-validation
//      Solver solver_;
      // testing size
      int testing_size_{0};
      std::vector< std::list<int> > testing_set_;
      // training size
      int training_size_{0};
      std::vector< std::list<int> > training_set_;
    };

  //
  //
  template< int K, typename M >
  CVKFolds< K, M >::CVKFolds()
    {
      //
      // folds construction
      std::size_t
	number_of_subjects = Alps::LoadDataSet::instance()->get_data()["inputs"]["images"][0].size();
      std::cout << "nombre subjects: " << number_of_subjects << std::endl;
      // check we don't have more folds than subjects
      if ( K > number_of_subjects )
	throw MAC::MACException( __FILE__, __LINE__,
				 "It can't be more folds than subjects.",
				 ITK_LOCATION );
      //
      testing_size_  = number_of_subjects / K;
      training_size_ = number_of_subjects - testing_size_;
      std::cout << "testing_size_: "  << testing_size_  << std::endl;
      std::cout << "training_size_: " << training_size_ << std::endl;
      // resize the sets
      testing_set_.resize( K );
      training_set_.resize( K );
      //
      int offset = 0;
      for ( int kk = 0 ; kk < K ; kk++ )
	{
	  for ( int s = 0 ; s < number_of_subjects ; s++ )
	    if ( s < testing_size_+offset && s >= offset )
	      testing_set_[kk].push_back(s);
	    else
	      training_set_[kk].push_back(s);
	  //
	  offset += testing_size_;
	}

      for ( int kk = 0 ; kk < K ; kk++ )
	{
	  for ( auto s = training_set_[kk].begin() ; s != training_set_[kk].end() ; s++ )
	    std::cout << " ~ " << *s << std::endl;
	  //
	  std::cout << std::endl;
	}
    }
  //
  //
  template< int K, typename M > void
  CVKFolds< K, M >::train()
    {}
  //
  //
  template< int K, typename M > void
  CVKFolds< K, M >::use()
    {}
}
#endif
