#include <random>
#include <Eigen/Sparse>
//
#include "Deconvolutional_window.h"
#include "MACLoadDataSet.h"



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
  // reverse the order of features
  number_of_features_in_  = Conv_wind->get_number_of_features_out();
  number_of_features_out_ = Conv_wind->get_number_of_features_in();

  //
  // Initialization of the weights
  //
  
  //
  // Weights
  number_of_weights_ = Conv_wind->get_number_of_weights();
  // Get the weights from the convolution window
  shared_weights_ = new double*[ number_of_features_in_ ];
  //
  for ( int feature = 0 ; feature < number_of_features_in_ ; feature++ )
    {
      shared_weights_[feature] = new double[ number_of_weights_ ];
      for ( int w = 0 ; w < number_of_weights_ ; w++ )
	shared_weights_[feature][w] = ( Conv_wind->get_shared_weights() )[feature][w];
    }
  // Initialize the biases
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution( -1.0, 1.0 );
  // initialization
  shared_biases_  = new double[ number_of_features_out_ ];
  //
  for ( std::size_t feature = 0 ; feature < number_of_features_out_ ; feature++ )
      shared_biases_[feature] = distribution(generator);
  

  //
  // Prepare the dimensions I/O
  // WARNING: information should be reverse from the convolution
  //

  //
  // Input dimension
  size_in_       = Conv_wind->get_size_out();
  origine_in_    = Conv_wind->get_origine_out();
  spacing_in_    = Conv_wind->get_spacing_out();
  direction_in_  = Conv_wind->get_direction_out();
  // Output dimensions
  size_out_      = Conv_wind->get_size_in();
  origine_out_   = Conv_wind->get_origine_in();
  spacing_out_   = Conv_wind->get_spacing_in();
  direction_out_ = Conv_wind->get_direction_in();
  //
  // Transfer the weight matrix
  im_size_in_    = size_in_[0]*size_in_[1]*size_in_[2];
  im_size_out_   = size_out_[0]*size_out_[1]*size_out_[2];
  //
  weights_poisition_oi_ = new std::size_t*[ im_size_out_ ];
  weights_poisition_io_ = new std::size_t*[ im_size_in_ ];
  //
  for ( std::size_t i = 0 ; i < im_size_in_ ; i++ )
    {
      weights_poisition_io_[i] = new std::size_t[number_of_weights_];
      for ( int k = 0 ; k < number_of_weights_ ; k++ )
	weights_poisition_io_[i][k] = ( Conv_wind->get_weights_position_oi() )[i][k];
    }
  //
  for ( std::size_t o = 0 ; o < im_size_out_ ; o++ )
    {
      weights_poisition_oi_[o] = new std::size_t[number_of_weights_];
      for ( int k = 0 ; k < number_of_weights_ ; k++ )
	weights_poisition_oi_[o][k] = ( Conv_wind->get_weights_position_io() )[o][k];
    }
}
//
//
void
MAC::Deconvolutional_window::print()
{}
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
