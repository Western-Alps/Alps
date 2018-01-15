//
//
//
#include <random>
#include "Convolutional_layer.h"

//
//
//
MAC::Convolutional_layer::Convolutional_layer( const std::string Layer_name,
					       const int         Layer_number,
					       const int         Shrinkage,
					       const bool        Do_we_pool,
					       const int*        Window_size ):
  MAC::NeuralNetwork::NeuralNetwork(),
  layer_name_{Layer_name}, layer_number_{Layer_number}, shrink_{Shrinkage}, pooling_operation_{Do_we_pool}
{
  //
  //
  memcpy ( convolution_window_size_, Window_size, 4*sizeof(int) );
  //
  // The window size is a 3+1 dimensions, the dimension must be odd! because we are
  // taking in account the center.
  // 3 dimensions for x,y and z; +1 dimension for the number of feature maps we would
  // like to create in this round.
  // To complete the kernel size, we need to know how many feature maps we had in the
  // previouse round.
  convolution_half_window_size_ = new int[4];
  //
  for ( int i = 0 ; i < 4 ; i++ )
    if ( convolution_window_size_[i] % 2 == 0 && i != 0 )
      {
	std::string mess = "The dimension of the window must be odd";
	mess += "dimension "  + std::to_string( i );
	mess += " value is: " + std::to_string( convolution_window_size_[i] );
	//
	throw MAC::MACException( __FILE__, __LINE__,
				 mess.c_str(),
				 ITK_LOCATION );
      }
    else
      {
	number_of_weights_ *= convolution_window_size_[i];
	// Half window's size
	if ( i > 0 )
	  convolution_half_window_size_[i] = static_cast< int >( (Window_size[i] - 1) / 2 );
	else
	  convolution_half_window_size_[i] = static_cast< int >( Window_size[i] );
      }

  //
  // Initialize the weights
  // we take the information from the pass round, on how many feature maps were created
  num_of_previous_features_ = static_cast< int >(MAC::Singleton::instance()->get_number_of_features());
  number_of_weights_ *= num_of_previous_features_;
  // add number of bias: biases of each kernel will be concataned at the end of the array
  number_of_weights_ += Window_size[0];
  // reset the number of feature map for the next round
  MAC::Singleton::instance()->set_number_of_features( Window_size[0] );
  //
  // Create the random weights
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution( -1.0, 1.0 );
  // initialization
  std::cout << "number_of_weights_ " << number_of_weights_ << std::endl;
  weights_ = new double[ number_of_weights_ ];
  for ( int w = 0 ; w < number_of_weights_ ; w++ )
    weights_[w] = distribution(generator);
  //
  convolution_images_.resize( Window_size[0] );
  pull_images_.resize( Window_size[0] );
};
//
//
//
void
MAC::Convolutional_layer::forward( Subject& Sub, const Weights& W )
{
  //
  // Convolution
  //

  //
  // Subject informations
  const std::vector< Image3DType::Pointer > curr_images = Sub.get_clone_modalities_images();
  std::string subject_name = Sub.get_subject_name();
  //
  // Images information
  Image3DType::IndexType  start = { 0, 0, 0 };
  Image3DType::Pointer    raw_subject_image_ptr = curr_images[0];
  Image3DType::SizeType   size = raw_subject_image_ptr->GetLargestPossibleRegion().GetSize();
  //
  Image3DType::PointType     orig_3d      = raw_subject_image_ptr->GetOrigin();
  Image3DType::SpacingType   spacing_3d   = raw_subject_image_ptr->GetSpacing();
  Image3DType::DirectionType direction_3d = raw_subject_image_ptr->GetDirection();
  //
  Image3DType::RegionType region;
  region.SetSize( size );
  region.SetIndex( start );

  //
  // Initialize the neurons, activation and delta
  if ( neurons_.find( subject_name ) == neurons_.end() )
    {
      std::vector< std::shared_ptr<double> >
	activations( convolution_window_size_[0] ),
	neurons( convolution_window_size_[0] ),
	deltas( convolution_window_size_[0] );
      //
      for ( int mod = 0 ; mod < convolution_window_size_[0] ; mod++ )
	{
	  size_t size_map = size[0] * size[1] * size[2];
	  activations[mod] = std::shared_ptr<double>( new double[size_map], std::default_delete< double[] >() );
	  neurons[mod]     = std::shared_ptr<double>( new double[size_map], std::default_delete< double[] >() );
	  deltas[mod]      = std::shared_ptr<double>( new double[size_map], std::default_delete< double[] >() );
	}
      //
      neurons_[subject_name] = std::make_tuple(activations,neurons,deltas);
    }

  //
  //
  for ( int mod = 0 ; mod < convolution_window_size_[0] ; mod++ )
    {
      //
      // Duplicate the image
      Image3DType::Pointer records = Image3DType::New();
      // image filter
      FilterType::Pointer images_filter;
      images_filter = FilterType::New();
      //
      images_filter->SetOutputSpacing( spacing_3d );
      images_filter->ChangeSpacingOn();
      images_filter->SetOutputOrigin( orig_3d );
      images_filter->ChangeOriginOn();
      images_filter->SetOutputDirection( direction_3d );
      images_filter->ChangeDirectionOn();
      //
      records->SetRegions( region );
      records->Allocate();
      records->FillBuffer( 0.0 );
      images_filter->SetInput( records );
      images_filter->Update();
      //
      convolution_images_[mod] = images_filter->GetOutput();

      //
      // Loop over the image
      itk::ImageRegionIterator< Image3DType > convolution_image_iter( convolution_images_[mod], region );
      //
      while( !convolution_image_iter.IsAtEnd() )
	{
	  //
	  //
	  Image3DType::IndexType idx = convolution_image_iter.GetIndex();
	  //
	  double convolution_voxel_value = 0;
	  //	  int X, x, Y, y, Z, z;
	  for ( int prev = 0 ; prev < num_of_previous_features_ ; prev++ ) // run through the num of previous features
	    for ( int z = - convolution_half_window_size_[3];
		  z < (convolution_half_window_size_[3]+1) ; z++ ) // run through z
	      for( int y = - convolution_half_window_size_[2];
		   y < (convolution_half_window_size_[2]+1) ; y++ ) // run through y
		for( int x = - convolution_half_window_size_[1];
		     x < (convolution_half_window_size_[1]+1) ; x++ ) // run through x
		  if ( idx[0] + x > -1 && idx[1] + y > -1 && idx[2] + z > -1 &&
		       idx[0] + x < static_cast<int>(size[0]) &&
		       idx[1] + y < static_cast<int>(size[1]) &&
		       idx[2] + z < static_cast<int>(size[2]) ) // zero padding
		    {
		      int weight_idx = (x+convolution_half_window_size_[1])
			+ convolution_window_size_[1] * (y+convolution_half_window_size_[2])
			+ convolution_window_size_[1] * convolution_window_size_[2]
			* (z+convolution_half_window_size_[3])
			+ convolution_window_size_[1] * convolution_window_size_[2]
			* convolution_window_size_[3] * prev
			+ convolution_window_size_[1] * convolution_window_size_[2]
			* convolution_window_size_[3] * num_of_previous_features_ * mod;
		      //
		      convolution_voxel_value +=
			weights_[ weight_idx ] * curr_images[prev]->GetPixel( {idx[0]+x, idx[1]+y, idx[2]+z} );
		    }
	  // add the bias at the end of the array
	  int bias_position = convolution_window_size_[0] * convolution_window_size_[1]
			    * convolution_window_size_[2] * convolution_window_size_[3]
			    * num_of_previous_features_ + mod;
	  convolution_voxel_value += weights_[ bias_position ];

	  //
	  // Update values
	  double
	     activation = convolution_voxel_value,
	     neuron     = tanh(convolution_voxel_value);
	  //
	  size_t image_position = idx[0] + size[0]*idx[1] + size[0]*size[1]*idx[2];
	  std::get< 0/*activations*/>(neurons_[subject_name])[mod].get()[image_position] = activation;
	  std::get< 1/*neurons*/>(neurons_[subject_name])[mod].get()[image_position] = neuron;
	  //
	  convolution_images_[mod]->SetPixel( idx, neuron );

	  //
	  // Iter the convolution image
	  ++convolution_image_iter;
	}
    }
  //
  write();

//  for ( int k = 0 ; k < size[2] ; k++ )
//    for ( int j = 0 ; j < size[1] ; j++ )
//      for ( int i = 0 ; i < size[0] ; i++ )
//	{
//	  std::cout << i << "," << j << "," << k << " --> ";
//	  size_t pos_temp = i + size[0] * j + size[0]*size[1]*k;
//	  std::cout << std::get< 1>(neurons_[subject_name])[0].get()[pos_temp] << std::endl;
//	}
  
  //
  // Pulling
  // Update the current image
  if ( pooling_operation_ )
    Sub.update( pulling() );
  else
    Sub.update( convolution_images_ );
};
//
//
//
const std::vector< Image3DType::Pointer > 
MAC::Convolutional_layer::pulling() 
{
  //
  //
  for ( int mod = 0 ; mod < convolution_window_size_[0] ; mod++ )
    {
      //
      // Shrink Image
      ShrinkImageFilterType::Pointer shrinkFilter
	= ShrinkImageFilterType::New();
      //
      shrinkFilter->SetInput( convolution_images_[mod]/*duplicator->GetOutput()*/ );
      // shrink the first dimension by a factor of 2
      shrinkFilter->SetShrinkFactor(0, shrink_);
      // shrink the first dimension by a factor of 2
      shrinkFilter->SetShrinkFactor(1, shrink_); 
      // shrink the first dimension by a factor of 2
      shrinkFilter->SetShrinkFactor(2, shrink_);
      //
      shrinkFilter->Update();
      //
      pull_images_[mod] = shrinkFilter->GetOutput();
      pull_images_[mod]->Update();
    }

  //
  //
  return pull_images_;
};
//
//
//
std::vector< Image3DType::SizeType > 
MAC::Convolutional_layer::pulling_image_size() 
{
  //
  //
  std::vector< Image3DType::SizeType > tempo( convolution_window_size_[0] );
  const std::vector< Image3DType::Pointer > curr_images =
    MAC::Singleton::instance()->get_subjects()[0].get_clone_modalities_images();
      
  //
  //
  for ( int mod = 0 ; mod < convolution_window_size_[0] ; mod++ )
    {
//      //
//      //
//      // Duplication the Convolutional image
//      DuplicatorType::Pointer duplicator = DuplicatorType::New();
//      duplicator->SetInputImage( curr_images[mod] );
//      duplicator->Update();

      //
      // Shrink Image
      ShrinkImageFilterType::Pointer shrinkFilter
	= ShrinkImageFilterType::New();
      //
      shrinkFilter->SetInput( curr_images[mod]/*duplicator->GetOutput()*/ );
      // shrink the first dimension by a factor of 2
      shrinkFilter->SetShrinkFactor(0, shrink_);
      // shrink the first dimension by a factor of 2
      shrinkFilter->SetShrinkFactor(1, shrink_); 
      // shrink the first dimension by a factor of 2
      shrinkFilter->SetShrinkFactor(2, shrink_);
      //
      shrinkFilter->Update();
      //
      tempo[mod] = shrinkFilter->GetOutput()->GetLargestPossibleRegion().GetSize();
    }

  //
  //
  return tempo;
};
//
//
//
MAC::Convolutional_layer::~Convolutional_layer() 
{
  delete [] weights_;
  weights_ = nullptr;
  delete [] convolution_window_size_;
  convolution_window_size_ = nullptr;
  delete [] convolution_half_window_size_;
  convolution_half_window_size_ = nullptr;
}
