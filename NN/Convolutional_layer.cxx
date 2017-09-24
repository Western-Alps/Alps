//
//
//
#include <random>
#include "Convolutional_layer.h"

/**
 * CUDA Kernel Device code
 *
 * Computes ...
 */
//__global__ void
//Convolutional_layer_backward_cuda( double *A, int numElements)
//{
//    int i = blockDim.x * blockIdx.x + threadIdx.x;
//
//    if (i < numElements)
//    {
//      
//        A[i] = A[i] * A[i];
//	/*printf("lalla %f", A[i]);*/
//    }
//}


//
//
//
MAC::Convolutional_layer::Convolutional_layer( const std::string Layer_name,
					       const int         Layer_number,
					       const int*        Window_size ):
  MAC::NeuralNetwork::NeuralNetwork(),
  layer_name_{Layer_name}, layer_number_{Layer_number}, 
  convolution_window_size_{Window_size}
{
  //
  // The window size is a 3 dimensions, the dimension must be odd!
  // because we are taking in account the center
  convolution_half_window_size_ = new int[3];
  //
  for ( int i = 0 ; i < 3 ; i++ )
    if ( convolution_window_size_[i] % 2 == 0 )
      {
	std::string mess = "The dimension of the half window must be odd";
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
	convolution_half_window_size_[i] = static_cast< int >( (Window_size[i] - 1) / 2 );
      }
  // We check how many modalities are in the data set
  num_of_modalities_ = static_cast< int >( MAC::Singleton::instance()->get_number_of_madalities() );
  // Adjust the number of weights to the number of modalities
  number_of_weights_ *= num_of_modalities_;
  //
  convolution_images_.resize( num_of_modalities_ );
  pull_images_.resize( num_of_modalities_ );
};
//
//
//
void
MAC::Convolutional_layer::forward( Subject& Sub, const Weights& W )
{

  //
  // Convolution
  const std::vector< Image3DType::Pointer > curr_images = Sub.get_clone_modalities_images();
  //
  for ( int mod = 0 ; mod < num_of_modalities_ ; mod++ )
    {
      //
      // Duplicate the image
      Image3DType::Pointer records = Image3DType::New();
      //
      Image3DType::RegionType region;
      Image3DType::IndexType  start = { 0, 0, 0 };
      Image3DType::Pointer    raw_subject_image_ptr = curr_images[mod];
      Image3DType::SizeType   size = raw_subject_image_ptr->GetLargestPossibleRegion().GetSize();
      //
      Image3DType::PointType     orig_3d      = raw_subject_image_ptr->GetOrigin();
      Image3DType::SpacingType   spacing_3d   = raw_subject_image_ptr->GetSpacing();
      Image3DType::DirectionType direction_3d = raw_subject_image_ptr->GetDirection();
      //
      region.SetSize( size );
      region.SetIndex( start );
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
      itk::ImageRegionIterator< Image3DType >
	orig_image_iter( curr_images[mod], region ),
	convolution_image_iter( convolution_images_[mod], region );
      //
      while( !orig_image_iter.IsAtEnd() )
	{
	  //
	  // Iter the convolution image
	  ++orig_image_iter;
	  ++convolution_image_iter;

	  //
	  //
	  Image3DType::IndexType idx = orig_image_iter.GetIndex();
	  //
	  std::vector< int > weight_indexes = W.get_weight_indexes();
	  const double*      weights        = W.get_weights();
	  //
	  int weight_idx = 0;
	  double convolution_voxel_value = 0;
	  for ( int z = idx[2] - convolution_half_window_size_[2] + 1;
		z < idx[2] + convolution_half_window_size_[2] ; z++ )
	    for( int y = idx[1] - convolution_half_window_size_[1] + 1;
		 y < idx[1] + convolution_half_window_size_[1] ; y++ )
	      for( int x = idx[0] - convolution_half_window_size_[0] + 1;
		   x < idx[0] + convolution_half_window_size_[0] ; x++ )
		if ( x > -1 && y > -1 && z > -1 &&
		     x < static_cast<int>(size[0]) && y < static_cast<int>(size[1]) && z < static_cast<int>(size[2]) )
		  {
		    convolution_voxel_value +=
		      weights[ weight_indexes[layer_number_]+weight_idx++ ] * curr_images[mod]->GetPixel( {x,y,z} );
		  }
		else
		  weight_idx++;
	  // Update value
	  convolution_images_[mod]->SetPixel( idx, convolution_voxel_value );
	}
    }
  //
  write();


  
  //
  // Pulling
  // Update the current image
  Sub.update( pulling() );
};
//
//
//
const std::vector< Image3DType::Pointer > 
MAC::Convolutional_layer::pulling() 
{
  //
  //
  for ( int mod = 0 ; mod < num_of_modalities_ ; mod++ )
    {
      //
      // Shrink Image
      ShrinkImageFilterType::Pointer shrinkFilter
	= ShrinkImageFilterType::New();
      //
      shrinkFilter->SetInput( convolution_images_[mod]/*duplicator->GetOutput()*/ );
      // shrink the first dimension by a factor of 2
      shrinkFilter->SetShrinkFactor(0, 2);
      // shrink the first dimension by a factor of 2
      shrinkFilter->SetShrinkFactor(1, 2); 
      // shrink the first dimension by a factor of 2
      shrinkFilter->SetShrinkFactor(2, 2);
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
  std::vector< Image3DType::SizeType > tempo( num_of_modalities_ );
  const std::vector< Image3DType::Pointer > curr_images =
    MAC::Singleton::instance()->get_subjects()[0].get_clone_modalities_images();
      
  //
  //
  for ( int mod = 0 ; mod < num_of_modalities_ ; mod++ )
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
      shrinkFilter->SetShrinkFactor(0, 2);
      // shrink the first dimension by a factor of 2
      shrinkFilter->SetShrinkFactor(1, 2); 
      // shrink the first dimension by a factor of 2
      shrinkFilter->SetShrinkFactor(2, 2);
      //
      shrinkFilter->Update();
      //
      tempo[mod] = shrinkFilter->GetOutput()->GetLargestPossibleRegion().GetSize();
    }

  //
  //
  return tempo;
};
