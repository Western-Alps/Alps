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
// Eigen
//
#include <Eigen/Sparse>
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
//#include "Convolutional_window.h"
//#include "Deconvolutional_window.h"
//
// CUDA
//
#include <cuda_runtime.h>

//
//
namespace MAC
{
  /** \class Convolution
   *
   * \brief 
   * 
   * 
   */
  template< class Grad, class ActivationFunction, class ConvDeconv  >
    class Convolution : public NeuralNetwork
  {

  public:

    /** Constructor. */
    Convolution( const std::string,
		 const int, 
		 ConvDeconv,
		 bool  Compute_cost_function_ = false );
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
    virtual void        backward();
    // Backward error propagation
    virtual void        backward_error_propagation(){};
    //
    // For the builders
    virtual void add( std::shared_ptr< NeuralNetwork > ){};
    // ToDo: do we need that?
    virtual int get_number_weights() const { return 0 /*number_of_weights_*/; };
    //
    // 
    void write( const std::string Subject_name ) const
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
	  std::string name = "Convolutional_" + Subject_name + "_" + layer_name_ + "_" + std::to_string(mod) + ".nii.gz";
	  writer->SetFileName( name );
	  writer->SetInput( img_ptr );
	  writer->SetImageIO( nifti_io );
	  writer->Update();
	  //
	  mod++;
	}
    };

  private:
    //
    // GPU treatment
    void forward_GPU( Subject& );
    void backward_GPU( );
    // CPU treatment
    void forward_CPU( Subject& );
    void backward_CPU( );

  protected:
    //
    // Inputs
    //
      
    // 
    // Convolution type
    Weight_type layer_type_{Unknown};
    // Convolutional layer's name
    std::string layer_name_;
    // layer energy (Cost function)
    double    energy_{0.};
    bool      compute_cost_function_{false};
    // Weights
    const int layer_number_;

    //
    // Convolution window
    ConvDeconv window_;

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
    // gradient
    // weights
    double** nabla_E_weights_{nullptr};
    // biases
    double*  nabla_E_biases_{nullptr};

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
  template< class G, class A, class CD >
    MAC::Convolution< G, A, CD >::Convolution( const std::string   Layer_name,
					       const int           Layer_number,
					       CD                  Window,
					       bool                Compute_cost_function_ ):
  MAC::NeuralNetwork::NeuralNetwork(),
    layer_name_{Layer_name}, compute_cost_function_{Compute_cost_function_}, layer_number_{Layer_number}, window_{Window}
			       
  {
    //
    layer_type_ = Window->get_layer_type();
    //
    int 
      num_kernels = 0,
      num_weights = 0;
    //
    switch( layer_type_ )
      {
      case Conv_layer:
	{
	  num_kernels = window_->get_number_of_features_out();
	  num_weights = window_->get_number_of_weights();
	  //
	  break;
	}
      case Deconv_layer:
	{
	  num_kernels = window_->get_number_of_features_in();
	  num_weights = window_->get_number_of_weights();
	  //
	  break;
	}
      default:
	{
	  std::string mess = "The type of layer is not defined for: ";
	  mess += layer_name_ + ".\n";
	  throw MAC::MACException( __FILE__, __LINE__,
				   mess.c_str(),
				   ITK_LOCATION );
	  
	}
      }
    //
    nabla_E_weights_ = new double*[ num_kernels ];
    nabla_E_biases_  = new double [ num_kernels ];
    //
    for ( int feature = 0 ; feature < num_kernels ; feature++ )
      {
	nabla_E_weights_[feature] = new double[ num_weights ];
	//
	for( int k = 0 ; k < num_weights ; k++ )
	  nabla_E_weights_[feature][k] = 0.;
	//
	nabla_E_biases_[feature] = 0.;
      }
  };
  //
  //
  //
  template< class G, class A, class CD > void
    MAC::Convolution< G, A, CD >::forward( Subject& Sub, const Weights& WW )
    {
      try
	{
	  //
	  //
	  if ( true /* CUDA */)
	    forward_GPU( Sub );
	  else     /*CPU*/
	    forward_CPU( Sub );
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	}
    };
  //
  //
  //
  template< class G, class A, class CD > void
    MAC::Convolution< G, A, CD >::forward_GPU( Subject& Sub )
    {
      try
	{
	  std::cout << "Go fwd GPU " << layer_name_ << " " << layer_type_ << std::endl;

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
	  // If costfunction calculation:
	  const std::vector< Image3DType::Pointer > tgt_images = Sub.get_modality_targets_ITK_images();
	  Image3DType::Pointer  subject_tgt_ptr = tgt_images[0];
	  Image3DType::SizeType size_tgt_       = subject_tgt_ptr->GetLargestPossibleRegion().GetSize();
	  int num_tgt_features                  = tgt_images.size();
	  //
	  std::size_t
	    im_size_prev,
	    im_size_next;

	  //
	  // Cuda treatment
	  Convolutional_CUDA cuda_treatment;

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
	  switch( layer_type_ )
	    {
	    case Conv_layer:
	      {
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
							window_->get_weights_position_io() );
		//
		break;
	      }
	    case Deconv_layer:
	      {
		cuda_treatment.load_deconvolution_kernels( // features
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
							  window_->get_weights_position_io() );
		//
		break;
	      }
	    default:
	      {
		std::string mess = "The type of layer is not defined for: ";
		mess += layer_name_ + ".\n";
		throw MAC::MACException( __FILE__, __LINE__,
					 mess.c_str(),
					 ITK_LOCATION );
		
	      }
	    }

	  
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
		  //Image3DType::IndexType idx = image_iter.GetIndex();
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
	  double** next_features_to_device   = new double*[num_of_next_features_];
	  double** next_activation_to_device = new double*[num_of_next_features_];
	  double** next_delta_to_device      = new double*[num_of_next_features_];
	  for ( int mod = 0 ; mod < num_of_next_features_ ; mod++ )
	    {
	      next_features_to_device[mod]   = new double[im_size_next];
	      next_activation_to_device[mod] = new double[im_size_next];
	      next_delta_to_device[mod]      = new double[im_size_next];
	    }
	  // Check we can compute the energy, if requiered
	  if ( compute_cost_function_ )
	    {
	      // Check the dim of the images
	      window_->check_match( size_out, size_tgt_ );
	      // 
	      if( num_tgt_features != num_of_next_features_ )
		{
		  std::string mess = "Cannot compute the Cost Function if the number of features out (";
		  mess += std::to_string( num_of_next_features_ ) + ") is different from the input ";
		  mess += "number of images (" + std::to_string( num_tgt_features ) + ").\n";
		  throw MAC::MACException( __FILE__, __LINE__,
					   mess.c_str(),
					   ITK_LOCATION );
		}
	    }   
	  //
	  // Convolution on GPU
	  switch( layer_type_ )
	    {
	    case Conv_layer:
	      {
		cuda_treatment.convolution( next_features_to_device,
					    next_activation_to_device,
					    next_delta_to_device,
					    activation_ );
		//
		break;
	      }
	    case Deconv_layer:
	      {
		cuda_treatment.transpose_convolution( next_features_to_device,
						      next_activation_to_device,
						      next_delta_to_device,
						      activation_ );
		//
		break;
	      }
	    default:
	      {
		std::string mess = "The type of layer is not defined for: ";
		mess += layer_name_ + ".\n";
		throw MAC::MACException( __FILE__, __LINE__,
					 mess.c_str(),
					 ITK_LOCATION );
		
	      }
	    }
	  
	  
	  //
	  // Write the image back
	  std::vector< std::shared_ptr<double> >
	    activations( num_of_next_features_ ),
	    neurons( num_of_next_features_ ),
	    deltas( num_of_next_features_ );
	  //
	  for ( int mod = 0 ; mod < num_of_next_features_ ; mod++ )
	    {
	      activations[mod] = std::shared_ptr<double>( new double[im_size_next], std::default_delete< double[] >() );
	      neurons[mod]     = std::shared_ptr<double>( new double[im_size_next], std::default_delete< double[] >() );
	      deltas[mod]      = std::shared_ptr<double>( new double[im_size_next], std::default_delete< double[] >() );
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
	      //
	      // Cost function calculation
	      int feature_idx = 0;
	      itk::ImageRegionIterator< Image3DType > convolution_iter( convolution_images_[mod], region_out );
	      if ( compute_cost_function_ )
		{
		  itk::ImageRegionIterator< Image3DType > tgd_iter( tgt_images[mod], region_out );
		  while( !convolution_iter.IsAtEnd() )
		    {
		      double local_img_energy = next_features_to_device[mod][feature_idx];
		      (activations[mod].get())[feature_idx] = local_img_energy;
		      // load the convolution
		      convolution_iter.Set( local_img_energy );
		      // Compute the local energy
		      local_img_energy -= tgd_iter.Value();
		      (deltas[mod].get())[feature_idx] = - local_img_energy * next_delta_to_device[mod][feature_idx];
		      energy_ += local_img_energy * local_img_energy;
		      ++convolution_iter; ++tgd_iter; ++feature_idx;
		    }
		}
	      else
		while( !convolution_iter.IsAtEnd() )
		  {
		    (activations[mod].get())[feature_idx] = next_features_to_device[mod][feature_idx];
		    (deltas[mod].get())[feature_idx]      = next_delta_to_device[mod][feature_idx];
		    convolution_iter.Set( next_features_to_device[mod][feature_idx++] );
		    ++convolution_iter; 
		  }
	      // clean up
	      delete [] next_features_to_device[mod];
	      delete [] next_activation_to_device[mod];
	      delete [] next_delta_to_device[mod];
	      next_features_to_device[mod]   = nullptr;
	      next_activation_to_device[mod] = nullptr;
	      next_delta_to_device[mod]      = nullptr;
	    }
	  //
	  ( window_->get_neuron() )[subject_name] = std::make_tuple(activations,neurons,deltas);

	  // clean up
	  delete [] next_features_to_device;
	  delete [] next_activation_to_device;
	  delete [] next_delta_to_device;
	  next_features_to_device   = nullptr;
	  next_activation_to_device = nullptr;
	  next_delta_to_device      = nullptr;
	  //
	  write( subject_name );
	  Sub.set_clone_modalities_images( convolution_images_ );
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	}
    };
  //
  //
  //
  template< class G, class A, class CD > void
    MAC::Convolution< G, A, CD >::forward_CPU( Subject& Sub )
    {
      try
	{
	  std::cout << "Go fwd CPU " << layer_name_ << " " << layer_type_ << std::endl;

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

	  //
	  switch( layer_type_ )
	    {
	    case Conv_layer:
	      {
		// 1. Load information
		num_of_previous_features_            = window_->get_number_of_features_in();
		num_of_next_features_                = window_->get_number_of_features_out();
		im_size_prev                         = window_->get_im_size_in();
		im_size_next                         = window_->get_im_size_out();
		Eigen::SparseMatrix< std::size_t > W = window_->get_W_out_in();
		std::vector< IOWeights > triplet_oiw     = window_->get_triplet_oiw();
		double**                 feature_weights = window_->get_shared_weights();               
		double*                  feature_biases  = window_->get_shared_biases();               
		//
		convolution_images_.resize( num_of_next_features_ );
		
		//
		// 2. Create the weights matrices
		std::vector< Eigen::SparseMatrix< double > > feature_W( num_of_next_features_ );
		// resize the matrix and allocate memory
		for ( int f = 0 ; f < num_of_next_features_ ; f++ )
		  {
		    feature_W[f].resize( im_size_next, im_size_prev );
		    feature_W[f].setFromTriplets( triplet_oiw.begin(), triplet_oiw.end(),
						  [] (const int&,const int& in) { return static_cast<double>(0.); } );
		  }
		// Create the W matrices for each features
		for ( int k = 0 ; k < W.outerSize() ; ++k )
		  for ( Eigen::SparseMatrix< std::size_t >::InnerIterator it( W, k ) ; it ; ++it )
		    for ( int f = 0 ; f < num_of_next_features_ ; f++ )
		      {
			feature_W[f].coeffRef( it.row(), it.col() ) = feature_weights[f][it.value()-1];
			//std::cout
			//	<< " feature: " << f
			//	<< " it.value() " << feature_weights[f][ it.value() - 1 ]
			//	<< " it.value() " << feature_W[f].coeff( it.row(), it.col() )
			//	<< " it.row(): "  << it.row()   // row index
			//	<< " it.col(): "  << it.col()   // col index 
			//	<< std::endl;
		      }
		
		//
		// 3. Convolution
		// 3.1. Load the images and create a vector
		std::vector< Eigen::VectorXd > prev_features_to_device( num_of_previous_features_ );
		// load the images into a vector
		for ( int prev = 0 ; prev < num_of_previous_features_ ; prev++ )
		  {
		    // ToDo: Is it the right order for z in [Z1,Zn]; y in [Y1,Yn]; x in [X1,Xn]
		    itk::ImageRegionIterator< Image3DType > image_iter( curr_images[prev], region );
		    prev_features_to_device[prev].resize( im_size_prev );
		    //
		    std::size_t current_position = 0;
		    while( !image_iter.IsAtEnd() )
		      {
			//
			prev_features_to_device[prev][ current_position++ ] = image_iter.Value();
			++image_iter;
		      }
		  }
		// 3.2. Create the new feature maps with convolution
		Image3DType::SizeType      size_out       = window_->get_size_out();
		Image3DType::PointType     origine_out    = window_->get_origine_out();
		Image3DType::SpacingType   spacing_out    = window_->get_spacing_out();
		Image3DType::DirectionType direction_out  = window_->get_direction_out();
		//
		Image3DType::RegionType region_out;
		region_out.SetSize( size_out );
		region_out.SetIndex( start );
		//
		std::vector< Eigen::VectorXd > next_features_to_device( num_of_next_features_ );
		for ( int mod = 0 ; mod < num_of_next_features_ ; mod++ )
		  {
		    next_features_to_device[mod].resize(im_size_next);
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
		    //
		    // Convolution
		    for ( int prev = 0 ; prev < num_of_previous_features_ ; prev++ )
		      next_features_to_device[mod] += feature_W[mod] * prev_features_to_device[prev];
		    //
		    // Create the output images
		    int feature_idx = 0;
		    itk::ImageRegionIterator< Image3DType > convolution_iter( convolution_images_[mod], region_out );
		    while( !convolution_iter.IsAtEnd() )
		      {
			// ToDo: Add the activation and the bias
			convolution_iter.Set( next_features_to_device[mod][feature_idx++] );
			++convolution_iter;
		      }
		  }
		//
		write( subject_name );
		Sub.set_clone_modalities_images( convolution_images_ );
		
		//
		//
		break;
	      }
	    case Deconv_layer:
	      {
		
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
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	}
    };
  //
  //
  //
  template< class G, class A, class CD > void
    MAC::Convolution< G, A, CD >::backward()
    {
      try
	{
	  //
	  //
	  if ( true /* CUDA */)
	    backward_GPU();
	  else     /*CPU*/
	    backward_CPU();
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	}
    };
  //
  //
  //
  template< class G, class A, class CD > void
    MAC::Convolution< G, A, CD >::backward_GPU()
    {
      try
	{
	  std::cout << "SIZE: " <<  ( window_->get_neuron() ).size() << std::endl;
	  //
	  //
	  std::size_t
	    im_size_prev,
	    im_size_next;

	  //
	  // Cuda treatment
	  Convolutional_CUDA cuda_treatment;

	  //
	  // 1. Load the data and initialize the GPU device
	  num_of_previous_features_ = window_->get_number_of_features_in();
	  num_of_next_features_     = window_->get_number_of_features_out();
	  im_size_prev              = window_->get_im_size_in();
	  im_size_next              = window_->get_im_size_out();
	  std::cout 
	    << "num_of_previous_features_ " << num_of_previous_features_
	    << "\n num_of_next_features_ " << num_of_next_features_
	    << "\n im_size_prev " << im_size_prev
	    << "\n im_size_next " << im_size_next
	    << std::endl;
	  // Check the dimensions of images with the window
	  window_->check_match( size_lu_, window_->get_size_in() );
	  // load the data
	  switch( layer_type_ )
	    {
	    case Conv_layer:
	      {
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
							window_->get_weights_position_io() );
		//
		break;
	      }
	    case Deconv_layer:
	      {
		cuda_treatment.load_deconvolution_kernels( // features
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
							  window_->get_weights_position_io() );
		//
		break;
	      }
	    default:
	      {
		std::string mess = "The type of layer is not defined for: ";
		mess += layer_name_ + ".\n";
		throw MAC::MACException( __FILE__, __LINE__,
					 mess.c_str(),
					 ITK_LOCATION );
		
	      }
	    }

	  //
	  // 2. For each subjects of the mini-batch run backwards
	  double** previous_features_to_device = nullptr;
	  double** delta_features_to_device    = nullptr;
	  for ( auto subject : window_->get_neuron() )
	    {
	      auto neuron = subject.second;
	      std::cout << "activations " << std::get< 0 /*activation*/ >(neuron).size() << std::endl;
	      std::cout << "neurons "     << std::get< 1 /*neurons*/    >(neuron).size() << std::endl;
	      std::cout << "deltas "      << std::get< 2 /*deltas*/     >(neuron).size() << std::endl;
	      
	      //
	      // If we have a previouse window we are in 
	      // - Auto-encoder deconvolution
	      // - Second convolution   (ToDo: check it works)
	      // - Second deconvolution (ToDo: check it works)
	      if ( window_->get_previouse_conv_window() )
		{
		  std::cout
		    << "LA PREVIOUSE: "
		    << std::get< 0/*activation*/>( window_->get_previouse_conv_window()
						   ->get_neuron()[subject.first] ).size()
		    <<std::endl;
		  //
		  // 2.1. load the vectors
		  // 2.1.1. Load the deltas
		  delta_features_to_device = new double*[num_of_next_features_];
		  for ( int mod = 0 ; mod < num_of_next_features_ ; mod++ )
		    {
		      //
		      delta_features_to_device[mod] = new double[im_size_next];
		      for ( std::size_t size = 0 ; size < im_size_next ; size++ )
			delta_features_to_device[mod][size] = 
			  std::get< 2/*deltas*/ >( window_->get_neuron()[subject.first] )[mod].get()[size];
		    }
		  // 2.1.2. load the previous features
		  previous_features_to_device = new double*[num_of_previous_features_];
		  for ( int mod = 0 ; mod < num_of_previous_features_ ; mod++ )
		    {
		      //
		      previous_features_to_device[mod] = new double[im_size_prev];
		      for ( std::size_t size = 0 ; size < im_size_prev ; size++ )
			previous_features_to_device[mod][size] = 
			  std::get< 0/*activation*/>( window_->get_previouse_conv_window()
						      ->get_neuron()[subject.first] )[mod].get()[size];
		    }
	      
		  //
		  // 2.2. Compute nabla and update the weights
		  cuda_treatment.backprog_transpose_convolution( delta_features_to_device, 
								 previous_features_to_device,
								 nabla_E_weights_, nabla_E_biases_);

		  
		  //
		  // 2.3. reset nabla
		  
		  
		  //
		  // 2.4. Clean up
		  for ( int mod = 0 ; mod < num_of_next_features_ ; mod++ )
		    {
		      delete [] delta_features_to_device[mod];
		      delta_features_to_device[mod] = nullptr;
		    }
		  // clean up
		  delete [] delta_features_to_device;
		  delta_features_to_device = nullptr;
		  
		  //
		  // 2.1.2. load the previous features
		  for ( int mod = 0 ; mod < num_of_previous_features_ ; mod++ )
		    {
		      delete [] previous_features_to_device[mod];
		      previous_features_to_device[mod] = nullptr;
		    }
		  // clean up
		  delete [] previous_features_to_device;
		  previous_features_to_device = nullptr;
		}
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
  template< class G, class A, class CD > void
    MAC::Convolution< G, A, CD >::backward_CPU()
    {
      try
	{
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	}
    };
  //
  //
  //
  template< class G, class A, class CD >
    MAC::Convolution< G, A, CD >::~Convolution() 
    {
    }
}
#endif
