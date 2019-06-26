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
