#include "Convolutional_window.h"
#include <random>



//
// Constructor
MAC::Convolutional_window::Convolutional_window(): Weights()
{}
//
// Constructor
MAC::Convolutional_window::Convolutional_window( const std::string Name,
						 const int* Conv_half_window,
						 const int* Striding,
						 const int* Padding,
						 const int  Num_of_features) : Weights(Name)
{
  //
  memcpy ( convolution_half_window_size_, Conv_half_window, 3*sizeof(int) );
  memcpy ( stride_, Striding, 3*sizeof(int) );
  memcpy ( padding_, Padding, 3*sizeof(int) );
  //
  number_of_features_ = Num_of_features;
}
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
//
//
int
MAC::Convolutional_window::feature_size( const int Dim ) const
{
  //
  // ToDo: a is the number of voxels in the 
  //       direction Dim
  int a = 120;
  // Output feature size in the direction Dim
  int feature_size = (a - 2*(convolution_half_window_size_[Dim] - padding_[Dim]) - 1);
  feature_size    /= stride_[Dim];


  //
  // ToDo: Check it is an integer
  return feature_size + 1;
}
//
//
MAC::Convolutional_window::~Convolutional_window()
{
  
}
