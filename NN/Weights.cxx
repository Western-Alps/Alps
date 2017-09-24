#include "Weights.h"
#include <random>



////
//// Allocating and initializing Weights's static data member.
//MAC::Weights* MAC::Weights::weights_instance_ = nullptr;
//
// Constructor
MAC::Weights::Weights():
  number_of_weights_{0}, weight_indexes_(), weights_{nullptr}
{}
//
// Constructor
MAC::Weights::Weights( const int Number_of_weights, const std::vector< int >  Weight_indexes ):
  number_of_weights_{Number_of_weights}, weight_indexes_(Weight_indexes)
{
  //
  // Create the random weights
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution( -1.0, 1.0 );
  //
  weights_ = new double[ number_of_weights_ ];
  for ( int w = 0 ; w < number_of_weights_  ; w++ )
    weights_[w] = distribution(generator);
}
//
//
void
MAC::Weights::print()
{
  //
  // check the number of weights
  std::cout << "number of weights: " << number_of_weights_ << std::endl;
  // Check the indexes:
  for ( auto u : weight_indexes_ )
    std::cout << "Indexes: " << u << std::endl;
  // Check the values of the weights
  for ( int w = 0 ; w < number_of_weights_  ; w++ )
    std::cout << weights_[w] << " ";
  //
  std::cout << std::endl;
}
