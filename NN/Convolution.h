#ifndef CONVOLUTION_H
#define CONVOLUTION_H
//
//
//
#include <iostream>
#include <fstream>
#include <math.h>  
#include <map>
#include <list>
#include <memory>
#include <random>
//
// CUDA
//
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include "Convolutional_CUDA.cuh"
//
// ITK
//
#include <itkSize.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkConstNeighborhoodIterator.h>
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
#include "Convolutional_window.h"
#include "Deconvolutional_window.h"
//
// CUDA
//
#include <cuda_runtime.h>

//
//
namespace MAC
{
  enum Convolution_type {Unknown, Conv_layer, Deconv_layer};
  /** \class Convolution
   *
   * \brief 
   * 
   * 
   */
  template< class Grad, class ActivationFunction  >
    class Convolution : public NeuralNetwork
    {

    public:

      /** Constructor. */
      explicit Convolution( const std::string,
			    const int, 
			    std::shared_ptr< Convolutional_window > );
      /** Constructor. */
      explicit Convolution( const std::string,
			    const int, 
			    std::shared_ptr< Deconvolutional_window > );
      /** Destructor */
      virtual ~Convolution();

      //
      // Initialization
      virtual void initialization(){};
      //
      // get the layer name
      virtual std::string get_layer_name(){ return layer_name_;};
      // get the layer type
      virtual Layer       get_layer_type(){ return convolution;};
      // get the energy
      virtual double      get_energy(){ return energy_;};
      // Forward propagation
      virtual void        forward( Subject&, const Weights& W = Weights() );
      // Backward propagation
      virtual void        backward(){};
      // Backward error propagation
      virtual void        backward_error_propagation(){};
      //
      // For the builders
      virtual void add( std::shared_ptr< NeuralNetwork > ){};
      // ToDo: do we need that?
      virtual int get_number_weights() const { return 0 /*number_of_weights_*/; };
      //
      // 
      void write() const
      {
	//
	// Check
	int mod = 0;
	for (auto img_ptr : convolution_images_ )
	  {
	    itk::NiftiImageIO::Pointer nifti_io = itk::NiftiImageIO::New();
	    //
	    itk::ImageFileWriter< Image3DType >::Pointer writer =
	      itk::ImageFileWriter< Image3DType >::New();
	    //
	    std::string name = "Convolutional_" + layer_name_ + "_" + std::to_string(mod) + ".nii.gz";
	    writer->SetFileName( name );
	    writer->SetInput( img_ptr );
	    writer->SetImageIO( nifti_io );
	    writer->Update();
	    //
	    mod++;
	  }
      };

    protected:
      //
      // Inputs
      //
      
      // 
      // Convolution type
      Convolution_type layer_type_{Unknown};
      // Convolutional layer's name
      std::string layer_name_;
      // layer energy
      double energy_{0.};
      // Weights
      const int  layer_number_;

      //
      // Convolution window
      std::shared_ptr< Convolutional_window >   window_;
      // Deconvolution window
      std::shared_ptr< Deconvolutional_window > dec_window_;

      //
      //
      int num_of_previous_features_{0};
      int num_of_next_features_{0};
      // Measures grouped in vector of 3D image
      // Convolution image: vector for each modality
      std::vector< Image3DType::Pointer > convolution_images_;

      //
      // Neurons, activations and delta
      std::map< std::string, std::tuple<
	std::vector< std::shared_ptr<double> > /* activations */,
	std::vector< std::shared_ptr<double> > /* neurons */,
	std::vector< std::shared_ptr<double> > /* deltas */ > > neurons_;

      //
      // Image information at the lu level
      Image3DType::SizeType      size_lu_;
      Image3DType::SpacingType   spacing_lu_;
      Image3DType::PointType     origine_lu_;
      Image3DType::DirectionType direction_lu_;

      //
      // Activation function
      ActivationFunction activation_;
      // Gradient method
      Grad               gradient_;

    };
  //
  //
  //
  // 
  //
  //
  //
  template< class G, class A >
    MAC::Convolution< G, A >::Convolution( const std::string   Layer_name,
					   const int           Layer_number,
					   std::shared_ptr< Convolutional_window > Window ):
  MAC::NeuralNetwork::NeuralNetwork(),
    layer_name_{Layer_name}, layer_number_{Layer_number}, window_{Window}, dec_window_{nullptr}
			       
  {
    layer_type_ = Conv_layer;
  };
  //
  //
  //
  template< class G, class A >
    MAC::Convolution< G, A >::Convolution( const std::string Layer_name,
					   const int         Layer_number,
					   std::shared_ptr< Deconvolutional_window > Window ):
  MAC::NeuralNetwork::NeuralNetwork(),
    layer_name_{Layer_name}, layer_number_{Layer_number}, window_{nullptr}, dec_window_{Window}
			       
  { 
    layer_type_ = Deconv_layer;
  };
  //
  //
  //
  template< class G, class A > void
    MAC::Convolution< G, A >::forward( Subject& Sub, const Weights& W )
    {
      try
	{
	  std::cout << "Go fwd " << layer_name_ << " " << layer_type_ << std::endl;

	  //
	  // Convolution
	  //

	  //
	  // Subject informations
	  const std::vector< Image3DType::Pointer > curr_images = Sub.get_clone_modalities_images();
	  std::string subject_name = Sub.get_subject_name();
	  std::cout << "subject_name " << subject_name << std::endl;
	  //
	  // Images information
	  Image3DType::IndexType  start = { 0, 0, 0 };
	  Image3DType::Pointer    raw_subject_image_ptr = curr_images[0];
	  //
	  size_lu_       = raw_subject_image_ptr->GetLargestPossibleRegion().GetSize();
	  origine_lu_    = raw_subject_image_ptr->GetOrigin();
	  spacing_lu_    = raw_subject_image_ptr->GetSpacing();
	  direction_lu_  = raw_subject_image_ptr->GetDirection();
	  //
	  Image3DType::RegionType region;
	  region.SetSize( size_lu_ );
	  region.SetIndex( start );
	  //
	  std::size_t
	    im_size_prev,
	    im_size_next;

	  
	  if ( true /* CUDA */)
	    {
	      //
	      // Cuda treatment
	      //
	      Convolutional_CUDA cuda_treatment;

	      switch( layer_type_ )
		{
		case Conv_layer:
		  {
		    //
		    // 1. Load the data and initialize the GPU device
		    num_of_previous_features_ = window_->get_number_of_features_in();
		    num_of_next_features_     = window_->get_number_of_features_out();
		    im_size_prev              = window_->get_im_size_in();
		    im_size_next              = window_->get_im_size_out();
		    convolution_images_.resize( num_of_next_features_ );
		    // Check the dimensions of images with the window
		    window_->check_match( size_lu_, window_->get_size_in() );
		    // load the data
		    cuda_treatment.load_convolution_kernels( // features
							    window_->get_number_of_features_in(),
							    window_->get_number_of_features_out(),
							    // weights
							    window_->get_number_of_weights(),
							    window_->get_shared_weights(),
							    window_->get_shared_biases(),
							    // weights position and transposed matrix
							    window_->get_im_size_in(),
							    window_->get_im_size_out(),
							    window_->get_weights_position_oi(),
							    window_->get_weights_position_io()/*,
											      // ToDo: to remove
											      window_->image_to_conv,
											      window_->image_conv*/ );
		    //
		    // 2. access the previouse feature maps and load it on the GPU device
		    double** prev_features_to_device = new double*[ num_of_previous_features_ ];
		    // Loop over the previous features image
		    for ( int prev = 0 ; prev < num_of_previous_features_ ; prev++ )
		      {
			// ToDo: Is it the right order for z in [Z1,Zn]; y in [Y1,Yn]; x in [X1,Xn]
			itk::ImageRegionIterator< Image3DType > image_iter( curr_images[prev], region );
			prev_features_to_device[prev] = new double[im_size_prev];
			//
			std::size_t current_position = 0;
			while( !image_iter.IsAtEnd() )
			  {
			    //
			    Image3DType::IndexType idx = image_iter.GetIndex();
			    prev_features_to_device[prev][ current_position++ ] = image_iter.Value();
			    //
			    ++image_iter;
			  }
		      }
		    // Load images on the device
		    cuda_treatment.load_feature_maps( prev_features_to_device );
		    // clean up
		    for ( int prev = 0 ; prev < num_of_previous_features_ ; prev++ )
		      {
			delete [] prev_features_to_device[prev];
			prev_features_to_device[prev] = nullptr;
		      }
		    delete [] prev_features_to_device;
		    prev_features_to_device = nullptr;

		    //
		    // 3. Create the new feature maps with convolution
		    Image3DType::SizeType      size_out       = window_->get_size_out();
		    Image3DType::PointType     origine_out    = window_->get_origine_out();
		    Image3DType::SpacingType   spacing_out    = window_->get_spacing_out();
		    Image3DType::DirectionType direction_out  = window_->get_direction_out();
		    //
		    Image3DType::RegionType region_out;
		    region_out.SetSize( size_out );
		    region_out.SetIndex( start );
		    //
		    double** next_features_to_device = new double*[num_of_next_features_];
		    for ( int mod = 0 ; mod < num_of_next_features_ ; mod++ )
		      {
			//
			//
			next_features_to_device[mod] = new double[im_size_next];
			
			//
			// Duplicate the image
			Image3DType::Pointer records = Image3DType::New();
			// image filter
			FilterType::Pointer images_filter;
			images_filter = FilterType::New();
			//
			images_filter->SetOutputSpacing( spacing_out );
			images_filter->ChangeSpacingOn();
			images_filter->SetOutputOrigin( origine_out );
			images_filter->ChangeOriginOn();
			images_filter->SetOutputDirection( direction_out );
			images_filter->ChangeDirectionOn();
			//
			records->SetRegions( region_out );
			records->Allocate();
			records->FillBuffer( 0.0 );
			images_filter->SetInput( records );
			images_filter->Update();
			//
			convolution_images_[mod] = images_filter->GetOutput();
		      }
		    
		    //
		    // Convolution on GPU
		    cuda_treatment.convolution( next_features_to_device, activation_ );
		    //
		    // Write the image back
		    for ( int mod = 0 ; mod < num_of_next_features_ ; mod++ )
		      {
			int feature_idx = 0;
			itk::ImageRegionIterator< Image3DType > convolution_iter( convolution_images_[mod], region_out );
			while( !convolution_iter.IsAtEnd() )
			  {
			    convolution_iter.Set( next_features_to_device[mod][feature_idx++] );
			    ++convolution_iter;
			  }
		      }
		    //
		    write();
		    Sub.set_clone_modalities_images( convolution_images_ );
		    //
		    //
		    break;
		  }
		case Deconv_layer:
		  {
		    //
		    // 1. Load the data and initialize the GPU device
		    num_of_previous_features_ = dec_window_->get_number_of_features_in();
		    num_of_next_features_     = dec_window_->get_number_of_features_out();
		    im_size_prev              = dec_window_->get_im_size_in();
		    im_size_next              = dec_window_->get_im_size_out();
		    convolution_images_.resize( num_of_next_features_ );
		    // Check the dimensions of images with the window
		    dec_window_->check_match( size_lu_, dec_window_->get_size_in() );
		    // load the data
		    cuda_treatment.load_deconvolution_kernels( // features
							      dec_window_->get_number_of_features_in(),
							      dec_window_->get_number_of_features_out(),
							      // weights
							      dec_window_->get_number_of_weights(),
							      dec_window_->get_shared_weights(),
							      dec_window_->get_shared_biases(),
							      // weights position and transposed matrix
							      dec_window_->get_im_size_in(),
							      dec_window_->get_im_size_out(),
							      dec_window_->get_weights_position_oi(),
							      dec_window_->get_weights_position_io() );
		    

		    //
		    // 2. access the previouse feature maps and load it on the GPU device
		    double** prev_features_to_device = new double*[ num_of_previous_features_ ];
		    // Loop over the previous features image
		    for ( int prev = 0 ; prev < num_of_previous_features_ ; prev++ )
		      {
			// ToDo: Is it the right order for z in [Z1,Zn]; y in [Y1,Yn]; x in [X1,Xn]
			itk::ImageRegionIterator< Image3DType > image_iter( curr_images[prev], region );
			prev_features_to_device[prev] = new double[im_size_prev];
			//
			std::size_t current_position = 0;
			while( !image_iter.IsAtEnd() )
			  {
			    //
			    Image3DType::IndexType idx = image_iter.GetIndex();
			    prev_features_to_device[prev][ current_position++ ] = image_iter.Value();
			    //
			    ++image_iter;
			  }
		      }
		    // Load images on the device
		    cuda_treatment.load_feature_maps( prev_features_to_device );
		    // clean up
		    for ( int prev = 0 ; prev < num_of_previous_features_ ; prev++ )
		      {
			delete [] prev_features_to_device[prev];
			prev_features_to_device[prev] = nullptr;
		      }
		    delete [] prev_features_to_device;
		    prev_features_to_device = nullptr;

		    //
		    // 3. Create the new feature maps with convolution
		    Image3DType::SizeType      size_out       = dec_window_->get_size_out();
		    Image3DType::PointType     origine_out    = dec_window_->get_origine_out();
		    Image3DType::SpacingType   spacing_out    = dec_window_->get_spacing_out();
		    Image3DType::DirectionType direction_out  = dec_window_->get_direction_out();
		    //
		    Image3DType::RegionType region_out;
		    region_out.SetSize( size_out );
		    region_out.SetIndex( start );
		    //
		    double** next_features_to_device = new double*[num_of_next_features_];
		    for ( int mod = 0 ; mod < num_of_next_features_ ; mod++ )
		      {
			//
			//
			next_features_to_device[mod] = new double[im_size_next];
			
			//
			// Duplicate the image
			Image3DType::Pointer records = Image3DType::New();
			// image filter
			FilterType::Pointer images_filter;
			images_filter = FilterType::New();
			//
			images_filter->SetOutputSpacing( spacing_out );
			images_filter->ChangeSpacingOn();
			images_filter->SetOutputOrigin( origine_out );
			images_filter->ChangeOriginOn();
			images_filter->SetOutputDirection( direction_out );
			images_filter->ChangeDirectionOn();
			//
			records->SetRegions( region_out );
			records->Allocate();
			records->FillBuffer( 0.0 );
			images_filter->SetInput( records );
			images_filter->Update();
			//
			convolution_images_[mod] = images_filter->GetOutput();
		      }
		    
		    //
		    // Convolution on GPU
		    cuda_treatment.transpose_convolution( next_features_to_device, activation_ );
		    //
		    // Write the image back
		    for ( int mod = 0 ; mod < num_of_next_features_ ; mod++ )
		      {
			int feature_idx = 0;
			itk::ImageRegionIterator< Image3DType > convolution_iter( convolution_images_[mod], region_out );
			while( !convolution_iter.IsAtEnd() )
			  {
			    convolution_iter.Set( next_features_to_device[mod][feature_idx++] );
			    ++convolution_iter;
			  }
		      }
		    //
		    write();
		    Sub.set_clone_modalities_images( convolution_images_ );
		    //
		    //
		    break;
		  }
		case Unknown:
		default:
		  {
		    std::string mess = "The type of layer is not defined for: ";
		    mess += layer_name_ + ".\n";
		    throw MAC::MACException( __FILE__, __LINE__,
					     mess.c_str(),
					     ITK_LOCATION );

		  }
		}
	    }
	  else /*CPU*/
	    {
	      // ToDo: Use the Eigen sparse Matrices system to validate CUDA
	    }
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	}
    };
  //
  //
  //
  template< class G, class A >
    MAC::Convolution< G, A >::~Convolution() 
    {
    }
}
#endif
