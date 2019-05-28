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
  // members
  memcpy ( convolution_half_window_size_, Conv_half_window, 3*sizeof(int) );
  memcpy ( stride_, Striding, 3*sizeof(int) );
  //memcpy ( padding_, {0,0,0}, 3*sizeof(int) );
  padding_ = nullptr; // ToDo padding dev
  //
  number_of_features_ = Num_of_features;

  //
  // Initialization of the weights
  int number_of_weights_ = 
    (2*(Conv_half_window[0]) + 1)*
    (2*(Conv_half_window[1]) + 1)*
    (2*(Conv_half_window[2]) + 1);
  // Create the random weights for each kernel
  // ToDo: Do the right initialization
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution( -1.0, 1.0 );
  // initialization
  shared_weights_ = new double*[ Num_of_features ];
  for ( int feature = 0 ; feature < Num_of_features ; feature++ )
    {
      shared_weights_[feature] = new double[ number_of_weights_ ];
      for ( int w = 0 ; w < number_of_weights_ ; w++ )
	shared_weights_[feature][w] = distribution(generator);
    }
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
  if (convolution_half_window_size_)
    delete [] convolution_half_window_size_;
  convolution_half_window_size_ = nullptr;
  if (stride_)
    delete [] stride_;
  stride_ = nullptr;
  if (padding_)
    delete [] padding_;
  padding_ = nullptr;
}
