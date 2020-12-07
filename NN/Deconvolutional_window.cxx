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
  // ToDo: Check Conv_wind exist
  
  //
  // members
  convolution_half_window_size_ = nullptr;
  stride_                       = nullptr;
  padding_                      = nullptr;
  //
  // reverse the order of features
  number_of_features_     = number_of_features_in_  = Conv_wind->get_number_of_features_out();
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
  for ( std::size_t feature = 0 ; feature < number_of_features_in_ ; feature++ )
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
  //std::cout << "In Deconv construc1" << std::endl;
  //std::cout << "size_in_ " << size_in_ << std::endl;
  //std::cout << "origine_in_ " << origine_in_ << std::endl;
  //std::cout << "spacing_in_ " << spacing_in_ << std::endl;
  //std::cout << "direction_in_ " << direction_in_ << std::endl;
  // Output dimensions
  size_out_      = Conv_wind->get_size_in();
  origine_out_   = Conv_wind->get_origine_in();
  spacing_out_   = Conv_wind->get_spacing_in();
  direction_out_ = Conv_wind->get_direction_in();
  //std::cout << "Out Deconv construc1" << std::endl;
  //std::cout << "size_out_ " << size_out_ << std::endl;
  //std::cout << "origine_out_ " << origine_out_ << std::endl;
  //std::cout << "spacing_out_ " << spacing_out_ << std::endl;
  //std::cout << "direction_out_ " << direction_out_ << std::endl;
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
       {
	 weights_poisition_io_[i][k] = ( Conv_wind->get_weights_position_oi() )[i][k];
	 //std::cout
	 //  << "weights_poisition_io_[" << i << "]["<< k << "] = " << weights_poisition_io_[i][k]
	 //  << std::endl;
       }
     }
 //
 for ( std::size_t o = 0 ; o < im_size_out_ ; o++ )
   {
     weights_poisition_oi_[o] = new std::size_t[number_of_weights_];
     for ( int k = 0 ; k < number_of_weights_ ; k++ )
       {
	 weights_poisition_oi_[o][k] = ( Conv_wind->get_weights_position_io() )[o][k];
	 //std::cout
	 //  << "weights_poisition_oi_[" << o << "]["<< k << "] = " << weights_poisition_oi_[o][k]
	 //  << std::endl;
       }
   }
}
//
//
//
void
MAC::Deconvolutional_window::print()
{}
//
//
//
void
MAC::Deconvolutional_window::check_match( ImageType<3>::SizeType Size_1,
					   ImageType<3>::SizeType Size_2 )
{
  try
    {
      if ( Size_1[0] != Size_2[0] ||
	   Size_1[1] != Size_2[1] ||
	   Size_1[2] != Size_2[2]  )
	{
	  std::string mess = "There is a dimension issue with the output: ";
	  mess += "( " + std::to_string( Size_1[0] ) ;
	  mess += ", " + std::to_string( Size_1[1] ) ;
	  mess += ", " + std::to_string( Size_1[2] ) + ") != ";
	  mess += "( " + std::to_string( Size_2[0] ) ;
	  mess += ", " + std::to_string( Size_2[1] ) ;
	  mess += ", " + std::to_string( Size_2[2] ) + ").";
	  //
	  throw MAC::MACException( __FILE__, __LINE__,
				   mess.c_str(),
				   ITK_LOCATION );
	}
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << err << std::endl;
      exit(-1);
    }
}
//
//
//
MAC::Deconvolutional_window::~Deconvolutional_window()
{
  //
  // Window
  if (convolution_half_window_size_)
    delete [] convolution_half_window_size_;
  convolution_half_window_size_ = nullptr;
  if (stride_)
    delete [] stride_;
  stride_ = nullptr;
  if (padding_)
    delete [] padding_;
  padding_ = nullptr;
  // Weights position matrices
  if (weights_poisition_oi_)
    {
      for (std::size_t o = 0 ; o < im_size_out_ ; o++)
	if ( weights_poisition_oi_[o] )
	  {
	    delete [] weights_poisition_oi_[o];
	    weights_poisition_oi_[o] = nullptr;
	  }
      delete [] weights_poisition_oi_;
      weights_poisition_oi_ = nullptr;
    }
  //
  if (weights_poisition_io_)
    {
      for (std::size_t i = 0 ; i < im_size_in_ ; i++)
	if ( weights_poisition_io_[i] )
	  {
	    delete [] weights_poisition_io_[i];
	    weights_poisition_io_[i] = nullptr;
	  }
      delete [] weights_poisition_io_;
      weights_poisition_io_ = nullptr;
    }
}
