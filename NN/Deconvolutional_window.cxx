#include "Deconvolutional_window.h"
#include <random>



//
// Constructor
MAC::Deconvolutional_window::Deconvolutional_window(): Weights()
{}
//
// Constructor
MAC::Deconvolutional_window::Deconvolutional_window( const std::string Name,
						     std::shared_ptr< MAC::Convolutional_window > Conv_wind ) : Weights(Name), previouse_conv_window_{Conv_wind}
{
  //
  // members
  convolution_half_window_size_ = nullptr;
  stride_                       = nullptr;
  padding_                      = nullptr;
  //
  number_of_features_in_  = Conv_wind->get_number_of_features_out();
  number_of_features_out_ = Conv_wind->get_number_of_features_in();

  //
  // Initialization of the weights
  //
  
  //
  int number_of_weights_ = Conv_wind->get_number_of_weights();
  // ToDo: here we will take the transposed matrix of the tensor
  // Initialize the biases
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution( -1.0, 1.0 );
  // initialization
  shared_weights_ = nullptr;
  shared_biases_  = new double[ number_of_features_out_ ];
  //
  for ( int feature = 0 ; feature < number_of_features_out_ ; feature++ )
      shared_biases_[feature] = distribution(generator);
  

  //
  // Prepare the dimensions I/O
  //

  //
  // Input dimension
  // ToDo: at this point we take only one input modality. Several input modalities
  //       would need to generalize the kernel to 4 dimensions
  size_in_       = Conv_wind->get_size_out();
  origine_in_    = Conv_wind->get_origine_out();
  spacing_in_    = Conv_wind->get_spacing_out();
  direction_in_  = Conv_wind->get_direction_out();
  //
  // Output dimensions
  size_out_       = Conv_wind->get_size_in();
  origine_out_    = Conv_wind->get_origine_in();
  spacing_out_    = Conv_wind->get_spacing_in();
  direction_out_  = Conv_wind->get_direction_in();
}
//
//
void
MAC::Deconvolutional_window::print()
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
MAC::Deconvolutional_window::~Deconvolutional_window()
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
