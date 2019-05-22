#include "Convolutional_window.h"
#include <random>



////
//// Allocating and initializing Convolutional_window's static data member.
//MAC::Convolutional_window* MAC::Convolutional_window::weights_instance_ = nullptr;
//
// Constructor
MAC::Convolutional_window::Convolutional_window(): Weights()
{}
//
// Constructor
MAC::Convolutional_window::Convolutional_window(const std::string Name ): Weights(Name)
{}
//
//
void
MAC::Convolutional_window::print()
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
