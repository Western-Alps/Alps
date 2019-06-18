#include <random>
//
#include "Convolutional_window.h"
#include "MACLoadDataSet.h"



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
						 const int  Num_of_features ) : Weights(Name), previouse_conv_window_{nullptr} // no previouse conv wind
{
  //
  // members
  convolution_half_window_size_ = new int[3];
  stride_                       = new int[3];
  padding_                      = new int[3];
  //
  for ( int d = 0 ; d < 3 ; d++ )
    {
      convolution_half_window_size_[d] = Conv_half_window[d];
      stride_[d]                       = Striding[d];
      padding_[d]                      = 0;
    }
  //memcpy ( convolution_half_window_size_, Conv_half_window, 3 * sizeof(int) );
  //memcpy ( stride_, Striding, 3 * sizeof(int) );
  //memcpy ( padding_, {0,0,0}, 3*sizeof(int) );
  //padding_ = nullptr; // ToDo padding dev
  //
  number_of_features_in_  = 1;
  number_of_features_out_ = Num_of_features;

  //
  // Initialization of the weights
  //
  
  // The dimensions of the window must be odd! We are taking in account the center.
  // 3 dimensions for x,y and z;
  // To complete the kernel size, we need to know how many feature maps we had in the
  // previouse round.
  //
  for ( int i = 0 ; i < 3 ; i++ )
    if ( Conv_half_window[i] % 2 == 0  )
      {
	std::string mess = "The dimension of the window must be odd";
	mess += " dimension " + std::to_string( i );
	mess += " value is: " + std::to_string( Conv_half_window[i] );
	//
	throw MAC::MACException( __FILE__, __LINE__,
				 mess.c_str(),
				 ITK_LOCATION );
      }
  //
  // WARNING: the bias represents index 0!
  number_of_weights_ = 
    (2*(Conv_half_window[0]) + 1)*
    (2*(Conv_half_window[1]) + 1)*
    (2*(Conv_half_window[2]) + 1);
  // Create the random weights for each kernel
  // ToDo: Do a better initialization
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution( -1.0, 1.0 );
  // initialization
  shared_weights_ = new double*[ Num_of_features ];
  shared_biases_  = new double[ Num_of_features ];
  //
  for ( int feature = 0 ; feature < Num_of_features ; feature++ )
    {
      shared_weights_[feature] = new double[ number_of_weights_ ];
      for ( int w = 0 ; w < number_of_weights_ ; w++ )
	shared_weights_[feature][w] = distribution(generator);
      //
      shared_biases_[feature] = distribution(generator);
    }
  

  //
  // Prepare the dimensions I/O
  //

  //
  // Input dimension
  // Take the first modality first image to get the dimensions
  // ToDo: at this point we take only one input modality. Several input modalities
  //       would need to generalize the kernel to 4 dimensions
  const std::vector< Image3DType::Pointer >
    curr_images = ( MAC::Singleton::instance()->get_subjects()[0] ).get_clone_modalities_images();
  //
  Image3DType::Pointer raw_subject_image_ptr = curr_images[0];
  size_in_       = raw_subject_image_ptr->GetLargestPossibleRegion().GetSize();
  origine_in_    = raw_subject_image_ptr->GetOrigin();
  spacing_in_    = raw_subject_image_ptr->GetSpacing();
  direction_in_  = raw_subject_image_ptr->GetDirection();
  std::cout << "In" << std::endl;
  std::cout << "size_in_ " << size_in_ << std::endl;
  std::cout << "origine_in_ " << origine_in_ << std::endl;
  std::cout << "spacing_in_ " << spacing_in_ << std::endl;
  std::cout << "direction_in_ " << direction_in_ << std::endl;
  //
  // Output dimensions
  size_out_       = feature_size( size_in_ );
  origine_out_    = feature_orig( size_in_, spacing_in_, origine_in_ );
  spacing_out_    = raw_subject_image_ptr->GetSpacing();
  direction_out_  = raw_subject_image_ptr->GetDirection();
  std::cout << "OUT" << std::endl;
  std::cout << "size_out_ " << size_out_ << std::endl;
  std::cout << "origine_out_ " << origine_out_ << std::endl;
  std::cout << "spacing_out_ " << spacing_out_ << std::endl;
  std::cout << "direction_out_ " << direction_out_ << std::endl;
}
//
//
// Constructor
MAC::Convolutional_window::Convolutional_window( const std::string Name,
						 std::shared_ptr< Convolutional_window > Conv_wind,
						 const int* Conv_half_window,
						 const int* Striding,
						 const int* Padding,
						 const int  Num_of_features ) : Weights(Name), previouse_conv_window_{Conv_wind}
{
  //
  // members
  convolution_half_window_size_ = new int[3];
  stride_                       = new int[3];
  padding_                      = new int[3];
  //
  for ( int d = 0 ; d < 3 ; d++ )
    {
      convolution_half_window_size_[d] = Conv_half_window[d];
      stride_[d]                       = Striding[d];
      padding_[d]                      = 0;
    }
  //
  number_of_features_in_  = Conv_wind->get_number_of_features_out();
  number_of_features_out_ = Num_of_features;

  //
  // Initialization of the weights
  //
  
  // The dimensions of the window must be odd! We are taking in account the center.
  // 3 dimensions for x,y and z;
  // To complete the kernel size, we need to know how many feature maps we had in the
  // previouse round.
  //
  for ( int i = 0 ; i < 3 ; i++ )
    if ( Conv_half_window[i] % 2 == 0  )
      {
	std::string mess = "The dimension of the window must be odd";
	mess += " dimension " + std::to_string( i );
	mess += " value is: " + std::to_string( Conv_half_window[i] );
	//
	throw MAC::MACException( __FILE__, __LINE__,
				 mess.c_str(),
				 ITK_LOCATION );
      }
  //
  // WARNING: the bias represents index 0!
  number_of_weights_ = 
    (2*(Conv_half_window[0]) + 1)*
    (2*(Conv_half_window[1]) + 1)*
    (2*(Conv_half_window[2]) + 1);
  // Create the random weights for each kernel
  // ToDo: Do a better initialization
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution( -1.0, 1.0 );
  // initialization
  shared_weights_ = new double*[ Num_of_features ];
  shared_biases_  = new double[ Num_of_features ];
  //
  for ( int feature = 0 ; feature < Num_of_features ; feature++ )
    {
      shared_weights_[feature] = new double[ number_of_weights_ ];
      for ( int w = 0 ; w < number_of_weights_ ; w++ )
	shared_weights_[feature][w] = distribution(generator);
      //
      shared_biases_[feature] = distribution(generator);
    }
  

  //
  // Prepare the dimensions I/O
  //

  //
  // Input dimension
  size_in_       = Conv_wind->get_size_out();	  
  origine_in_    = Conv_wind->get_origine_out();  
  spacing_in_    = Conv_wind->get_spacing_out();  
  direction_in_  = Conv_wind->get_direction_out();
  std::cout << "In" << std::endl;
  std::cout << "size_in_ " << size_in_ << std::endl;
  std::cout << "origine_in_ " << origine_in_ << std::endl;
  std::cout << "spacing_in_ " << spacing_in_ << std::endl;
  std::cout << "direction_in_ " << direction_in_ << std::endl;
  //
  // Output dimensions
  size_out_       = feature_size( size_in_ );
  origine_out_    = feature_orig( size_in_, spacing_in_, origine_in_ );
  spacing_out_    = Conv_wind->get_spacing_out();
  direction_out_  = Conv_wind->get_direction_out();
  std::cout << "OUT" << std::endl;
  std::cout << "size_out_ " << size_out_ << std::endl;
  std::cout << "origine_out_ " << origine_out_ << std::endl;
  std::cout << "spacing_out_ " << spacing_out_ << std::endl;
  std::cout << "direction_out_ " << direction_out_ << std::endl;
}
//
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
//
void
MAC::Convolutional_window::save_weights()
{
  //
  //
  std::ofstream out( name_, std::ios::out | std::ios::binary | std::ios::trunc );
  //
  out.write( (char*) (&number_of_features_in_),  sizeof(size_t) );
  out.write( (char*) (&number_of_features_out_), sizeof(size_t) );
  //
  for ( std::size_t feature = 0 ; feature < number_of_features_out_ ; feature++ )
    {
      out.write( (char*) shared_weights_[feature], number_of_weights_*sizeof(double) );
      out.write( (char*) &shared_biases_[feature], sizeof(double) );
    }
}
//
//
//
void
MAC::Convolutional_window::load_weights()
{
  //
  //
  std::ifstream out( name_, std::ios::in | std::ios::binary );
  //
  out.read( (char*) (&number_of_features_in_),  sizeof(size_t) );
  out.read( (char*) (&number_of_features_out_), sizeof(size_t) );
  //
  for ( std::size_t feature = 0 ; feature < number_of_features_out_ ; feature++ )
    {
      out.read( (char*) shared_weights_[feature], number_of_weights_*sizeof(double) );
      out.read( (char*) &shared_biases_[feature], sizeof(double) );
    }
}
//
//
Image3DType::SizeType
MAC::Convolutional_window::feature_size( const Image3DType::SizeType Input_size ) const
{
  //
  // Output feature size in the direction Dim
  Image3DType::SizeType to_return;
  //
  for ( int d = 0 ; d < 3 ; d++ )
    {
      to_return[d]  = ( Input_size[d] - 2 * convolution_half_window_size_[d] - 1 + 2 * padding_[d]);
      to_return[d] /= stride_[d];
      to_return[d] +=  1;
    }      

  //
  //
  return to_return;
}
//
//
Image3DType::PointType
MAC::Convolutional_window::feature_orig( const Image3DType::SizeType    Input_size,
					 const Image3DType::SpacingType Input_spacing,
					 const Image3DType::PointType   Input_orig ) const
{
  //
  // Output feature size in the direction Dim
  Image3DType::SizeType  size;
  Image3DType::PointType to_return;
  //
  for ( int d = 0 ; d < 3 ; d++ )
    {
      size[d]  = ( Input_size[d] - 2 * convolution_half_window_size_[d] - 1 + 2 * padding_[d]);
      size[d] /= stride_[d];
      size[d] +=  1;
      //
      to_return[d] = 0.5 * ( size[d] - 1 ) * Input_spacing[d];
      to_return[d] = ( Input_orig[d] > 0 ? to_return[d] : -1 * to_return[d] );
    }      

  //
  //
  return to_return;
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
