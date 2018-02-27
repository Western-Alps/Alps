#ifndef CONVOLUTIONALAUTOENCODER_LAYER_H
#define CONVOLUTIONALAUTOENCODER_LAYER_H
//
//
//
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <math.h>  
#include <map>
#include <list>
#include <memory>
#include <random>
//
// ITK
//
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>
#include <itkNiftiImageIO.h>
#include <itkOrientImageFilter.h>
#include <itkSpatialOrientation.h>
#include "itkChangeInformationImageFilter.h"
#include "itkImageDuplicator.h"
// {down,up}sampling the pooling image and upsampling
#include "itkIdentityTransform.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkShrinkImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkReLUInterpolateImageFunction.h"

//
// Some typedef
using Image3DType    = itk::Image< double, 3 >;
using Reader3D       = itk::ImageFileReader< Image3DType >;
using Writer3D       = itk::ImageFileWriter< Image3DType >;
using MaskType       = itk::Image< unsigned char, 3 >;
using FilterType     = itk::ChangeInformationImageFilter< Image3DType >;
using DuplicatorType = itk::ImageDuplicator< Image3DType > ;
using ShrinkImageFilterType = itk::ShrinkImageFilter < Image3DType, Image3DType >;
//
//
//
#include "MACException.h"
#include "MACLoadDataSet.h"
#include "Subject.h"
#include "NeuralNetwork.h"
#include "Weights.h"
//
// CUDA
//
#include <cuda_runtime.h>

//
//
namespace MAC
{
  /** \class ConvolutionalAutoEncoder_layer
   *
   * \brief 
   * 
   * 
   */
  template< class ActivationFunction  >
    class ConvolutionalAutoEncoder_layer : public Convolutional_layer< ActivationFunction >
    {

    public:

      /** Constructor encoder. */
      explicit ConvolutionalAutoEncoder_layer( const std::string,
					       const int, 
					       const int,
					       const int* );
      /** Constructor decoder. */
      explicit ConvolutionalAutoEncoder_layer( const std::string,
					       const int,
					       std::shared_ptr< MAC::NeuralNetwork > );
      /** Destructor */
      virtual ~ConvolutionalAutoEncoder_layer();

      //
      // Initialization
      virtual void initialization(){};
      //
      // get the layer name
      virtual std::string get_layer_name(){ return Convolutional_layer< ActivationFunction >::get_layer_name();};
      //
      // get the layer name
      virtual Layer get_layer_type(){ return layer_type_;};
      //
      // Forward propagation
      virtual void forward( Subject&, const Weights& W = Weights() );
      //
      //
      virtual void backward(){};
      //
      // Backward error propagation
      virtual void backward_error_propagation(){};
      //
      //
      virtual void add( std::shared_ptr< NeuralNetwork > ){};
      //
      //
      virtual int get_number_weights() const{return Convolutional_layer< ActivationFunction >::get_number_weights();};

    private:
      //
      //
      Layer layer_type_;
      //
      int     number_of_decode_weights_{1};
      double* weights_T_{nullptr};

      //
      //
      Convolutional_layer< ActivationFunction >* encoder_conv_layer_{NULL};
      
      //
      // Activation function
      ActivationFunction activation_;
      };
  //
  //
  //
  // 
  //
  //
  //
  /*
    Layer_name,   
    Layer_number
    Shrinkage: if it is positive it downsample, otherwise it up sample,  
    Do_we_pool, 
    Window_size 
  */
  template< class A >
    MAC::ConvolutionalAutoEncoder_layer< A >::ConvolutionalAutoEncoder_layer( const std::string Layer_name,
									      const int         Layer_number,
									      const int         Shrinkage,
									      const int*        Window_size ):
  MAC::Convolutional_layer< A >::Convolutional_layer( Layer_name, Layer_number, Shrinkage, Window_size ),
    layer_type_{ convolutional_encoder }
  {
  };
  //
  //
  //
  /*
    Layer_name,   
    Layer_number
    Window_size 
  */
  template< class A >
    MAC::ConvolutionalAutoEncoder_layer< A >::ConvolutionalAutoEncoder_layer( const std::string Layer_name,
									      const int         Layer_number,
									      std::shared_ptr< MAC::NeuralNetwork > Encoder ):
  MAC::Convolutional_layer< A >::Convolutional_layer( Layer_name,
						      Layer_number,
						      dynamic_cast< Convolutional_layer<A>* >( Encoder.get() )->get_window_size() ),
    layer_type_{ convolutional_decoder },
    encoder_conv_layer_{ dynamic_cast< Convolutional_layer<A>* >( Encoder.get() )}
  {
    //
    // 1. reset the weights in weights_T_
    // One bias per chanels. The bias will be at the end of the serries of weights
    int* coding_window = encoder_conv_layer_->get_window_size();
    //
    int coding_weights_number = coding_window[0]
      * Convolutional_layer<A>::get_window_size()[1]
      * Convolutional_layer<A>::get_window_size()[2]
      * Convolutional_layer<A>::get_window_size()[3];
    int decoding_bias_weights_number = Convolutional_layer<A>::get_window_size()[0];
    //
    number_of_decode_weights_ = coding_weights_number + decoding_bias_weights_number;
    //
    weights_T_ = new double[ number_of_decode_weights_ ];
    // copy the coding weights into the pointer
    memcpy ( weights_T_, encoder_conv_layer_->get_weights(),
	     coding_weights_number * sizeof(double) );
    // create new weights for each channels
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution( -1.0, 1.0 );
    for ( int w = 0 ; w < decoding_bias_weights_number ; w++ )
      weights_T_[w+coding_weights_number] = distribution(generator);
  };
  //
  //
  //
  template< class A > void
    MAC::ConvolutionalAutoEncoder_layer< A >::forward( Subject& Sub, const Weights& W )
    {
      try
	{
	  switch( layer_type_ )
	    {
	    case convolutional_encoder:
	      {
		Convolutional_layer< A >::forward( Sub );
		break;
	      }
	    case convolutional_decoder:
	      {
		//
		// Subject informations
		const std::vector< Image3DType::Pointer > curr_images   = Sub.get_clone_modalities_images();
		const std::vector< Image3DType::Pointer > target_images = Sub.get_modality_targets_ITK_images();
		std::string subject_name = Sub.get_subject_name();
		//
		// Images information
		Image3DType::IndexType  start = { 0, 0, 0 };
		Image3DType::Pointer    raw_subject_image_ptr = curr_images[0];
		//
		Convolutional_layer<A>::size_lu_       = raw_subject_image_ptr->GetLargestPossibleRegion().GetSize();
		Convolutional_layer<A>::origine_lu_    = raw_subject_image_ptr->GetOrigin();
		Convolutional_layer<A>::spacing_lu_    = raw_subject_image_ptr->GetSpacing();
		Convolutional_layer<A>::direction_lu_  = raw_subject_image_ptr->GetDirection();
		//
		Image3DType::RegionType region;
		region.SetSize( Convolutional_layer<A>::size_lu_ );
		region.SetIndex( start );
		//
		std::size_t neuron_number = Convolutional_layer<A>::size_lu_[0]*Convolutional_layer<A>::size_lu_[1]*Convolutional_layer<A>::size_lu_[2];

		//
		// Check the images comply
		auto size_target       = target_images[0]->GetLargestPossibleRegion().GetSize();;
		auto origine_target    = target_images[0]->GetOrigin();
		auto spacing_target    = target_images[0]->GetSpacing();
		auto direction_target  = target_images[0]->GetDirection();
		//
		if ( Convolutional_layer<A>::size_lu_      != size_target    ||
		     Convolutional_layer<A>::origine_lu_   != origine_target ||
		     Convolutional_layer<A>::spacing_lu_   != spacing_target ||
		     Convolutional_layer<A>::direction_lu_ != direction_target )
		  throw MAC::MACException( __FILE__, __LINE__,
					   "Feature maps and taget map must have identical size, origine, spacing and direction. \n Check no pooling is applied.",
					   ITK_LOCATION );

		  
		//
		//
		//
		// 1. Init the CUDA device and the new feature output
		//    Cuda device must be already initialized from the coding phase
		//    We will initialize the weights_T
		encoder_conv_layer_->get_cuda().set_weights_T( number_of_decode_weights_,
							       weights_T_ );

		//
		// Initialize the neurons, activation and delta
		if ( Convolutional_layer<A>::neurons_.find( subject_name ) == Convolutional_layer<A>::neurons_.end() )
		  {
		    std::vector< std::shared_ptr<double> >
		      activations( Convolutional_layer<A>::convolution_window_size_[0] ),
		      neurons( Convolutional_layer<A>::convolution_window_size_[0] ),
		      deltas( Convolutional_layer<A>::convolution_window_size_[0] );
		    //
		    for ( int mod = 0 ; mod < Convolutional_layer<A>::convolution_window_size_[0] ; mod++ )
		      {
			activations[mod] = std::shared_ptr<double>( new double[neuron_number], std::default_delete< double[] >() );
			neurons[mod]     = std::shared_ptr<double>( new double[neuron_number], std::default_delete< double[] >() );
			deltas[mod]      = std::shared_ptr<double>( new double[neuron_number], std::default_delete< double[] >() );
		      }
		    //
		    Convolutional_layer<A>::neurons_[subject_name] = std::make_tuple(activations,neurons,deltas);
		  }

	  
		//
		// 2. access the previouse feature maps and target maps
		double** prev_features_to_device;
		double** target_features_to_device;
		Mapping* prev_idx_mapping_to_device;
		//
		prev_features_to_device      = new double*[ curr_images.size() ];
		prev_idx_mapping_to_device   = new Mapping[ neuron_number ];
		target_features_to_device    = new double*[ target_images.size() ];
		// Loop over the previouse features maps
		for ( int prev = 0 ; prev < curr_images.size() ; prev++ ) // run through the previous features
		  {
		    prev_features_to_device[prev]    = new double[ neuron_number ];
		    //
		    itk::ImageRegionIterator< Image3DType > convolution_image_iter( curr_images[prev], region );
		    std::size_t current_position = 0;
		    while( !convolution_image_iter.IsAtEnd() )
		      {
			//
			Image3DType::IndexType idx = convolution_image_iter.GetIndex();
			//
			if  ( prev == 0 )
			  {
			    prev_idx_mapping_to_device[current_position].idx_ = current_position;
			    prev_idx_mapping_to_device[current_position].x_   = idx[0];
			    prev_idx_mapping_to_device[current_position].y_   = idx[1];
			    prev_idx_mapping_to_device[current_position].z_   = idx[2];
			  }
			//
			prev_features_to_device[prev][current_position++] = convolution_image_iter.Value();
			//
			++convolution_image_iter;
		      }
		  }
		//
		for ( int targ = 0 ; targ < target_images.size() ; targ++ ) // run through the target images
		  {
		    target_features_to_device[targ]    = new double[ neuron_number ];
		    //
		    itk::ImageRegionIterator< Image3DType > convolution_image_iter( target_images[targ], region );
		    std::size_t current_position = 0;
		    while( !convolution_image_iter.IsAtEnd() )
		      {
			//
			Image3DType::IndexType idx = convolution_image_iter.GetIndex();
			//
			target_features_to_device[targ][current_position++] = convolution_image_iter.Value();
			//
			++convolution_image_iter;
		      }
		  }

		//
		// Load images on the device
		encoder_conv_layer_->get_cuda().load_previouse_feature_maps( prev_features_to_device,
									     target_features_to_device,
									     prev_idx_mapping_to_device,
									     curr_images.size(),
									     target_images.size(),
									     neuron_number );

		//
		// 3. Create the new feature maps with convolution
		for ( int mod = 0 ; mod < Convolutional_layer<A>::convolution_window_size_[0] ; mod++ )
		  {	    
		    //
		    // Duplicate the image
		    Image3DType::Pointer records = Image3DType::New();
		    // image filter
		    FilterType::Pointer images_filter;
		    images_filter = FilterType::New();
		    //
		    images_filter->SetOutputSpacing( Convolutional_layer<A>::spacing_lu_ );
		    images_filter->ChangeSpacingOn();
		    images_filter->SetOutputOrigin( Convolutional_layer<A>::origine_lu_ );
		    images_filter->ChangeOriginOn();
		    images_filter->SetOutputDirection( Convolutional_layer<A>::direction_lu_ );
		    images_filter->ChangeDirectionOn();
		    //
		    records->SetRegions( region );
		    records->Allocate();
		    records->FillBuffer( 0.0 );
		    images_filter->SetInput( records );
		    images_filter->Update();
		    //
		    Convolutional_layer<A>::convolution_images_[mod] = images_filter->GetOutput();

		    //
		    // Convolution on GPU
		    encoder_conv_layer_->get_cuda().convolution_decoding( Convolutional_layer<A>::neurons_[subject_name],
									  mod, activation_ );
		    //
		    itk::ImageRegionIterator< Image3DType > convolution_iter( Convolutional_layer<A>::convolution_images_[mod],
									      region );
		    int feature_idx = 0;
		    while( !convolution_iter.IsAtEnd() )
		      {
			convolution_iter.Set( (std::get< 1/*neurons*/>( Convolutional_layer<A>::neurons_[subject_name] ))[mod].get()[feature_idx++] );
			++convolution_iter;
		      }
		  }
		//
		//
		Convolutional_layer<A>::write();
		
		//  for ( int k = 0 ; k < size_lu_[2] ; k++ )
		//    for ( int j = 0 ; j < size_lu_[1] ; j++ )
		//      for ( int i = 0 ; i < size_lu_[0] ; i++ )
		//	{
		//	  std::cout << i << "," << j << "," << k << " --> ";
		//	  size_t pos_temp = i + size_lu_[0] * j + size_lu_[0]*size_lu_[1]*k;
		//	  std::cout << std::get< 1>(neurons_[subject_name])[0].get()[pos_temp] << std::endl;
		//	}
		
		//
		// Pulling
		// Update the current image
		if ( Convolutional_layer<A>::pooling_operation_ )
		  Sub.update( Convolutional_layer<A>::resample() );
		else if ( Convolutional_layer<A>::match_inputs_ )
		  Sub.update( Convolutional_layer<A>::reconstruct_inputs( Sub ) /*resample( Sub )*/ );
		else
		  Sub.update( Convolutional_layer<A>::convolution_images_ );
		
		//
		//
		break;
	      }
	    default:
	      throw MAC::MACException( __FILE__, __LINE__,
				       "this function can only be used for stacked convolutional auto-encoder.",
				       ITK_LOCATION );
	    }
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  exit(-1);
	}
    };
  //
  //
  //
  template< class A >
    MAC::ConvolutionalAutoEncoder_layer< A >::~ConvolutionalAutoEncoder_layer() 
    {
    }
}
#endif
