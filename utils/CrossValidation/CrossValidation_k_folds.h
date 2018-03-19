#ifndef CROSSVALIDATION_K_FOLDS_H
#define CROSSVALIDATION_K_FOLDS_H
//
//
//
#include <iostream>
#include <fstream>
//
//
//
#include "MACException.h"
#include "CrossValidation.h"
#include "MACLoadDataSet.h"
//
//
//
namespace MAC
{
  /** \class CrossValidation_k_folds
   *
   * \brief 
   * 
   */
  template< typename Solver >
    class CrossValidation_k_folds : public CrossValidation
    {
    public:
      /** Constructor. */
    explicit CrossValidation_k_folds();
      
      /**  */
      virtual ~CrossValidation_k_folds( )
	{};

      //
      // train the calssification engin
      virtual void train();
      // use the calssification engin
      virtual void use();

    private:
      //
      // Solver used in the cross-validation
      Solver solver_;
      // number of folds
      int k_{3};
      // testing size
      int testing_size_{0};
      std::vector< std::list<int> > testing_set_;
      // training size
      int training_size_{0};
      std::vector< std::list<int> > training_set_;
    };

  //
  //
  template< typename Solver >
    CrossValidation_k_folds< Solver >::CrossValidation_k_folds():
  CrossValidation()
    {
      //
      // GPU interogation

      //
      // folds construction
      std::size_t number_of_subjects = MAC::Singleton::instance()->get_subjects().size();
      std::cout << "nombre subjects: " << number_of_subjects << std::endl;
      // check we don't have more folds than subjects
      if ( k_ > number_of_subjects )
	throw MAC::MACException( __FILE__, __LINE__,
				 "It can't be more folds than subjects.",
				 ITK_LOCATION );
      //
      testing_size_  = number_of_subjects / k_;
      training_size_ = number_of_subjects - testing_size_;
      std::cout << "testing_size_: " << testing_size_ << std::endl;
      std::cout << "training_size_: " << training_size_ << std::endl;
      // resize the sets
      testing_set_.resize( k_ );
      training_set_.resize( k_ );
      //
      int offset = 0;
      for ( int kk = 0 ; kk < k_ ; kk++ )
	{
	  for ( int s = 0 ; s < number_of_subjects ; s++ )
	    if ( s < testing_size_+offset && s >= offset )
	      testing_set_[kk].push_back(s);
	    else
	      training_set_[kk].push_back(s);
	  //
	  offset += testing_size_;
	}
//
//      for ( int kk = 0 ; kk < k_ ; kk++ )
//	{
//	  for ( auto s = training_set_[kk].begin() ; s != training_set_[kk].end() ; s++ )
//	    std::cout << " ~ " << *s << std::endl;
//	  //
//	  std::cout << std::endl;
//	}
    }
  //
  //
  template< typename Solver > void
    CrossValidation_k_folds< Solver >::train()
    {}
  //
  //
  template< typename Solver > void
    CrossValidation_k_folds< Solver >::use()
    {}
}
#endif
