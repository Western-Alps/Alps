#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H
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
#include "Convolutional_layer_CUDA.cuh"
// ITK
#include "ITKHeaders.h"
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

  /** \class Convolutional_layer
   *
   * \brief 
   * 
   * 
   */
  template< class ActivationFunction  >
    class Convolutional_layer : public NeuralNetwork
    {

    public:

      /** Constructor. */
      explicit Convolutional_layer( const std::string,
				    const int, 
				    const int,
				    const int* );
      /** Constructor. 
	  In the case of Mont Blanc technics, we need a constructor to match the 
	  input images
      */
      explicit Convolutional_layer( const std::string,
				    const int, 
				    const int* );
      /** Destructor */
      virtual ~Convolutional_layer();

      //
      // Initialization
      virtual void initialization(){};
      //
      // get the layer name
      virtual std::string get_layer_name(){ return layer_name_;};
      //
      // get the layer name
      virtual Layer get_layer_type(){ return convolutional_layer;};
      //
      // get the layer name
      virtual double get_energy(){ return energy_;};
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
      virtual int get_number_weights() const { return number_of_weights_; };
      //
      //
      int get_shrink() const { return shrink_; };
      //
      //
      int* get_window_size() const { return convolution_window_size_; };
      //
      //
      double* get_weights(){ return weights_; };
      //
      //
      Convolutional_layer_CUDA& get_cuda(){ return cuda_treatment_; };
      //
      //
      const std::vector< ImageType<3>::Pointer > pooling();
      //
      //
      std::vector< ImageType<3>::SizeType > pooling_image_size();
      //
      //
      const std::vector< ImageType<3>::Pointer > resample();
      //
      //
      const std::vector< ImageType<3>::Pointer > resample( const Subject& );
      //
      //
      const std::vector< ImageType<3>::Pointer > resample_backward();
      //
      //
      const std::vector< ImageType<3>::Pointer > reconstruct_inputs( const Subject& );
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
	    itk::ImageFileWriter< ImageType<3> >::Pointer writer =
	      itk::ImageFileWriter< ImageType<3> >::New();
	    //
	    std::string name = "convolutional_" + layer_name_ + "_" + std::to_string(mod) + ".nii.gz";
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
      // Convolutional layer's name
      std::string layer_name_;
      // layer energy
      double energy_{0.};
      
      //
      // Weights
      const int  layer_number_;
      // Shrinkage for the pooling layer
      int        shrink_;
      bool       pooling_operation_{true};
      bool       match_inputs_{false};
      int        number_of_weights_{1};
      double*    weights_{nullptr};
      
      //
      // Size of the convolution window
      int* convolution_window_size_{new int[4]};
      int* convolution_half_window_size_;


      //
      //
      int num_of_previous_features_{0};
      // Measures grouped in vector of 3D image
      // Convolution image: vector for each modality
      std::vector< ImageType<3>::Pointer > convolution_images_;
      // Pulling image: vector for each modality
      std::vector< ImageType<3>::Pointer > pull_images_;

      //
      // Neurons, activations and delta
      std::map< std::string, std::tuple<
	std::vector< std::shared_ptr<double> > /* activations */,
	std::vector< std::shared_ptr<double> > /* neurons */,
	std::vector< std::shared_ptr<double> > /* deltas */ > > neurons_;

      //
      // Image information at the lu level
      ImageType<3>::SizeType      size_lu_;
      ImageType<3>::SpacingType   spacing_lu_;
      ImageType<3>::PointType     origine_lu_;
      ImageType<3>::DirectionType direction_lu_;

      //
      // Activation function
      ActivationFunction activation_;

      //
      // Cuda treatment
      Convolutional_layer_CUDA cuda_treatment_;
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
    MAC::Convolutional_layer< A >::Convolutional_layer( const std::string Layer_name,
							const int         Layer_number,
							const int         Shrinkage,
							const int*        Window_size ):
  MAC::NeuralNetwork::NeuralNetwork(),
    layer_name_{Layer_name}, layer_number_{Layer_number}, shrink_{Shrinkage}, pooling_operation_{true}
  {
    //
    // If shrinkage is 0: we don't pool
    if ( Shrinkage == 0 )
      pooling_operation_ = false;
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
  /*
    Layer_name,   
    Layer_number
    Window_size 
  */
  template< class A >
    MAC::Convolutional_layer< A >::Convolutional_layer( const std::string Layer_name,
							const int         Layer_number,
							const int*        Window_size ):
  MAC::NeuralNetwork::NeuralNetwork(),
    layer_name_{Layer_name}, layer_number_{Layer_number}, shrink_{0}, pooling_operation_{false},match_inputs_{true}
  {
    //
    //
    memcpy ( convolution_window_size_, Window_size, 4*sizeof(int) );
    // Get the number of target modalities, must be the same as the number of input images
    int number_of_input_modalities = static_cast< int >(MAC::Singleton::instance()->get_number_of_input_features());
    // replace the input number of feature maps
    convolution_window_size_[0] = number_of_input_modalities;
  
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
	    convolution_half_window_size_[i] = number_of_input_modalities;
	}

    //
    // Initialize the weights
    // we take the information from the pass round, on how many feature maps were created
    num_of_previous_features_ = static_cast< int >(MAC::Singleton::instance()->get_number_of_features());
    //to rm  number_of_weights_ *= num_of_previous_features_;
    // add number of bias: biases of each kernel will be concataned at the end of the array
    number_of_weights_ += number_of_input_modalities;
    // reset the number of feature map for the next round
    MAC::Singleton::instance()->set_number_of_features( number_of_input_modalities );
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
    convolution_images_.resize( number_of_input_modalities );
    pull_images_.resize( number_of_input_modalities );
  };
  //
  //
  //
  template< class A > void
    MAC::Convolutional_layer< A >::forward( Subject& Sub, const Weights& W )
    {
      //
      // Convolution
      //

      //
      // Subject informations
      const std::vector< ImageType<3>::Pointer > curr_images = Sub.get_clone_modalities_images();
      std::string subject_name = Sub.get_subject_name();
      //
      // Images information
      ImageType<3>::IndexType  start = { 0, 0, 0 };
      ImageType<3>::Pointer    raw_subject_image_ptr = curr_images[0];
      //
      size_lu_       = raw_subject_image_ptr->GetLargestPossibleRegion().GetSize();
      origine_lu_    = raw_subject_image_ptr->GetOrigin();
      spacing_lu_    = raw_subject_image_ptr->GetSpacing();
      direction_lu_  = raw_subject_image_ptr->GetDirection();
      //
      ImageType<3>::RegionType region;
      region.SetSize( size_lu_ );
      region.SetIndex( start );
      //
      std::size_t neuron_number = size_lu_[0]*size_lu_[1]*size_lu_[2];


      //
      //
      if ( true /*CUDA*/)
	{
	  //
	  // 1. Init the CUDA device and the new feature output
	  int image_size[3] = {static_cast<int>(size_lu_[0]),
			       static_cast<int>(size_lu_[1]),
			       static_cast<int>(size_lu_[2])};
	  //
	  cuda_treatment_.init( image_size,
				convolution_half_window_size_,
				number_of_weights_, weights_ );

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
		  activations[mod] = std::shared_ptr<double>( new double[neuron_number], std::default_delete< double[] >() );
		  neurons[mod]     = std::shared_ptr<double>( new double[neuron_number], std::default_delete< double[] >() );
		  deltas[mod]      = std::shared_ptr<double>( new double[neuron_number], std::default_delete< double[] >() );
		}
	      //
	      neurons_[subject_name] = std::make_tuple(activations,neurons,deltas);
	    }



	  
	  //
	  // 2. access the previouse feature maps and load it on the GPU device
	  double** prev_features_to_device;
	  Mapping* prev_idx_mapping_to_device;
	  //
	  prev_features_to_device    = new double*[ curr_images.size() ];
	  prev_idx_mapping_to_device = new Mapping[ neuron_number ];
	  // Loop over the image
	  for ( int prev = 0 ; prev < num_of_previous_features_ ; prev++ ) // run through the previous features
	    {
	      prev_features_to_device[prev]    = new double[ neuron_number ];
	      //
	      itk::ImageRegionIterator< ImageType<3> > convolution_image_iter( curr_images[prev], region );
	      std::size_t current_position = 0;
	      while( !convolution_image_iter.IsAtEnd() )
		{
		  //
		  ImageType<3>::IndexType idx = convolution_image_iter.GetIndex();
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
	  // Load images on the device
	  cuda_treatment_.load_previouse_feature_maps( prev_features_to_device,
						       prev_idx_mapping_to_device,
						       num_of_previous_features_,
						       neuron_number );

	  //
	  // 3. Create the new feature maps with convolution
	  for ( int mod = 0 ; mod < convolution_window_size_[0] ; mod++ )
	    {	    
	      //
	      // Duplicate the image
	      ImageType<3>::Pointer records = ImageType<3>::New();
	      // image filter
	      FilterType<3>::Pointer images_filter;
	      images_filter = FilterType<3>::New();
	      //
	      images_filter->SetOutputSpacing( spacing_lu_ );
	      images_filter->ChangeSpacingOn();
	      images_filter->SetOutputOrigin( origine_lu_ );
	      images_filter->ChangeOriginOn();
	      images_filter->SetOutputDirection( direction_lu_ );
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
	      // Convolution on GPU
	      cuda_treatment_.convolution( neurons_[subject_name],
					   mod, activation_ );
	      //
	      itk::ImageRegionIterator< ImageType<3> > convolution_iter( convolution_images_[mod], region );
	      int feature_idx = 0;
	      while( !convolution_iter.IsAtEnd() )
		{
		  convolution_iter.Set( (std::get< 1/*neurons*/>( neurons_[subject_name] ))[mod].get()[feature_idx++] );
		  ++convolution_iter;
		}
	    }
	}
      else
	{
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
		  size_t size_map = size_lu_[0] * size_lu_[1] * size_lu_[2];
		  activations[mod] = std::shared_ptr<double>( new double[size_map], std::default_delete< double[] >() );
		  neurons[mod]     = std::shared_ptr<double>( new double[size_map], std::default_delete< double[] >() );
		  deltas[mod]      = std::shared_ptr<double>( new double[size_map], std::default_delete< double[] >() );
		}
	      //
	      neurons_[subject_name] = std::make_tuple(activations,neurons,deltas);
	    }

	  //
	  // Create the new feature maps
	  for ( int mod = 0 ; mod < convolution_window_size_[0] ; mod++ )
	    {
	      //
	      // Duplicate the image
	      ImageType<3>::Pointer records = ImageType<3>::New();
	      // image filter
	      FilterType<3>::Pointer images_filter;
	      images_filter = FilterType<3>::New();
	      //
	      images_filter->SetOutputSpacing( spacing_lu_ );
	      images_filter->ChangeSpacingOn();
	      images_filter->SetOutputOrigin( origine_lu_ );
	      images_filter->ChangeOriginOn();
	      images_filter->SetOutputDirection( direction_lu_ );
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
	      itk::ImageRegionIterator< ImageType<3> > convolution_image_iter( convolution_images_[mod], region );
	      //
	      while( !convolution_image_iter.IsAtEnd() )
		{
		  //
		  //
		  ImageType<3>::IndexType idx = convolution_image_iter.GetIndex();
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
			       idx[0] + x < static_cast<int>(size_lu_[0]) &&
			       idx[1] + y < static_cast<int>(size_lu_[1]) &&
			       idx[2] + z < static_cast<int>(size_lu_[2]) ) // zero padding
			    {
			      int weight_idx = (x+convolution_half_window_size_[1])
				+ convolution_window_size_[1] * (y+convolution_half_window_size_[2])
				+ convolution_window_size_[1] * convolution_window_size_[2]
				* (z+convolution_half_window_size_[3])
				+ convolution_window_size_[1] * convolution_window_size_[2]
				* convolution_window_size_[3] * mod;
			      //
			      convolution_voxel_value +=
				weights_[ weight_idx ] * curr_images[prev]->GetPixel( {idx[0]+x, idx[1]+y, idx[2]+z} );
			    }
		  // add the bias at the end of the array
		  int bias_position = convolution_window_size_[0] * convolution_window_size_[1]
		    * convolution_window_size_[2] * convolution_window_size_[3]
		    + mod;
		  convolution_voxel_value += weights_[ bias_position ]; // x 1.
	  
		  //
		  // Update values
		  double
		    activation = convolution_voxel_value,
		    neuron     = activation_.f( convolution_voxel_value );
		  //
		  size_t image_position = idx[0] + size_lu_[0]*idx[1] + size_lu_[0]*size_lu_[1]*idx[2];
		  std::get< 0/*activations*/>(neurons_[subject_name])[mod].get()[image_position] = activation;
		  std::get< 1/*neurons*/>(neurons_[subject_name])[mod].get()[image_position]     = neuron;
		  //
		  convolution_images_[mod]->SetPixel( idx, neuron );

		  //
		  // Iter the convolution image
		  ++convolution_image_iter;
		}
	    }
	}
      //
      // write();

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
       if ( pooling_operation_ )
       	Sub.update( resample() );
       else if ( match_inputs_ )
       	Sub.update( reconstruct_inputs( Sub ) /*resample( Sub )*/ );
       else
       	Sub.update( convolution_images_ );
    };
  //
  //
  //
  template< class A > const std::vector< ImageType<3>::Pointer > 
    MAC::Convolutional_layer< A >::pooling() 
    {
      //
      //
      for ( int mod = 0 ; mod < convolution_window_size_[0] ; mod++ )
	{
	  //
	  // Shrink Image
	  ShrinkImageFilterType<3>::Pointer shrinkFilter
	    = ShrinkImageFilterType<3>::New();
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
  template< class A > std::vector< ImageType<3>::SizeType > 
    MAC::Convolutional_layer< A >::pooling_image_size() 
    {
      //
      //
      std::vector< ImageType<3>::SizeType > tempo( convolution_window_size_[0] );
      const std::vector< ImageType<3>::Pointer > curr_images =
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
	  ShrinkImageFilterType<3>::Pointer shrinkFilter
	    = ShrinkImageFilterType<3>::New();
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
  template< class A > const std::vector< ImageType<3>::Pointer > 
    MAC::Convolutional_layer< A >::resample() 
    {
      //
      //
      for ( int mod = 0 ; mod < convolution_window_size_[0] ; mod++ )
	{
	  //
	  // Images information
	  ImageType<3>::Pointer    raw_subject_image_ptr = convolution_images_[mod];
	  ImageType<3>::SizeType   size = raw_subject_image_ptr->GetLargestPossibleRegion().GetSize();
	  //
	  ImageType<3>::PointType     orig_3d      = raw_subject_image_ptr->GetOrigin();
	  //ImageType<3>::SpacingType   spacing_3d   = raw_subject_image_ptr->GetSpacing();
	  ImageType<3>::DirectionType direction_3d = raw_subject_image_ptr->GetDirection();

	  //
	  // Resize
	  ImageType<3>::SizeType outputSize;
	  ImageType<3>::SpacingType outputSpacing;
	  //
	  for ( int i = 0 ; i < 3 ; i++ )
	    {
	      if ( shrink_ < 0 )
		outputSize[i] = -1 * size[i] * shrink_;
	      else
		outputSize[i] = static_cast< int >( size[i] / shrink_ );
	      //
	      outputSpacing[i] = raw_subject_image_ptr->GetSpacing()[i] * (static_cast<double>(size[i]) / static_cast<double>(outputSize[i]));
	    }
	  //
	  typedef itk::IdentityTransform< double, 3 > TransformType;
	  typedef itk::LinearInterpolateImageFunction< ImageType<3>, double >  InterpolatorType;
	  //typedef itk::ReLUInterpolateImageFunction< ImageType<3>, double >  InterpolatorType;
	  typedef itk::ResampleImageFilter< ImageType<3>, ImageType<3> > ResampleImageFilterType;
	  //
	  InterpolatorType::Pointer interpolator    = InterpolatorType::New();
	  ResampleImageFilterType::Pointer resample = ResampleImageFilterType::New();
	  //
	  resample->SetInput(raw_subject_image_ptr);
	  resample->SetInterpolator( interpolator );
	  resample->SetSize(outputSize);
	  resample->SetOutputSpacing(outputSpacing);
	  resample->SetOutputOrigin(orig_3d);
	  resample->SetOutputDirection(direction_3d);
	  resample->SetTransform(TransformType::New());
	  resample->UpdateLargestPossibleRegion();
      
	  //
	  //
	  pull_images_[mod] = resample->GetOutput();
	  pull_images_[mod]->Update();
	}

      //
      //
      return pull_images_;
    };
  //
  //
  //
  template< class A > const std::vector< ImageType<3>::Pointer > 
    MAC::Convolutional_layer< A >::resample( const Subject& Sub ) 
    {
      //
      //
      for ( int mod = 0 ; mod < convolution_window_size_[0] ; mod++ )
	{
	  //
	  // Images layer information information
	  ImageType<3>::Pointer    raw_subject_image_ptr = convolution_images_[mod];
	  //ImageType<3>::SizeType   size = raw_subject_image_ptr->GetLargestPossibleRegion().GetSize();
	  //
	  ImageType<3>::PointType     orig_3d      = raw_subject_image_ptr->GetOrigin();
	  //ImageType<3>::SpacingType   spacing_3d   = raw_subject_image_ptr->GetSpacing();
	  ImageType<3>::DirectionType direction_3d = raw_subject_image_ptr->GetDirection();
	  //
	  // Images input information information
	  ImageType<3>::Pointer input_ptr = Sub.get_modality_targets_ITK_images()[mod];
	  //
	  ImageType<3>::SizeType      in_ptr_size         = input_ptr->GetLargestPossibleRegion().GetSize();
	  //ImageType<3>::PointType     in_ptr_orig_3d      = input_ptr->GetOrigin();
	  ImageType<3>::SpacingType   in_ptr_spacing_3d   = input_ptr->GetSpacing();
	  ImageType<3>::DirectionType in_ptr_direction_3d = input_ptr->GetDirection();

	  //
	  // Resize
	  typedef itk::IdentityTransform<double, 3> TransformType;
	  typedef itk::ResampleImageFilter<ImageType<3>, ImageType<3>> ResampleImageFilterType;
	  ResampleImageFilterType::Pointer resample = ResampleImageFilterType::New();
	  //
	  resample->SetInput(raw_subject_image_ptr);
	  resample->SetSize(in_ptr_size);
	  resample->SetOutputSpacing(in_ptr_spacing_3d);
	  resample->SetOutputOrigin(orig_3d);
	  resample->SetOutputDirection(direction_3d);
	  resample->SetTransform(TransformType::New());
	  resample->UpdateLargestPossibleRegion();
      
	  //
	  //
	  pull_images_[mod] = resample->GetOutput();
	  pull_images_[mod]->Update();
	}

      //
      //
      return pull_images_;
    };
  //
  //
  //
  template< class A > const std::vector< ImageType<3>::Pointer > 
    MAC::Convolutional_layer< A >::resample_backward() 
    {
      //
      //
      for ( int mod = 0 ; mod < convolution_window_size_[0] ; mod++ )
	{
	  //
	  // Images layer information information
	  ImageType<3>::Pointer    raw_subject_image_ptr = pull_images_[mod];
	  //ImageType<3>::SizeType   size = raw_subject_image_ptr->GetLargestPossibleRegion().GetSize();
	  //
	  ImageType<3>::PointType     orig_3d      = raw_subject_image_ptr->GetOrigin();
	  //ImageType<3>::SpacingType   spacing_3d   = raw_subject_image_ptr->GetSpacing();
	  ImageType<3>::DirectionType direction_3d = raw_subject_image_ptr->GetDirection();

	  //
	  // Resize
	  typedef itk::IdentityTransform<double, 3> TransformType;
	  typedef itk::ResampleImageFilter<ImageType<3>, ImageType<3>> ResampleImageFilterType;
	  ResampleImageFilterType::Pointer resample = ResampleImageFilterType::New();
	  //
	  resample->SetInput(raw_subject_image_ptr);
	  resample->SetSize(size_lu_);
	  resample->SetOutputSpacing(spacing_lu_);
	  resample->SetOutputOrigin(orig_3d);
	  resample->SetOutputDirection(direction_3d);
	  resample->SetTransform(TransformType::New());
	  resample->UpdateLargestPossibleRegion();
      
	  //
	  //
	  convolution_images_[mod] = resample->GetOutput();
	  convolution_images_[mod]->Update();
	}

      //
      //
      return convolution_images_;
    };
  //
  //
  //
  template< class A > const std::vector< ImageType<3>::Pointer > 
    MAC::Convolutional_layer< A >::reconstruct_inputs( const Subject& Sub ) 
    {
      //
      // Resample to the original input dimensions
      resample( Sub );
    
      //
      // Create the delta image between the decoding phase and the inputs
      for ( int mod = 0 ; mod < convolution_window_size_[0] ; mod++ )
	{
	  //
	  // Images layer information information
	  ImageType<3>::Pointer    raw_subject_image_ptr = pull_images_[mod];
	  ImageType<3>::SizeType   size = raw_subject_image_ptr->GetLargestPossibleRegion().GetSize();
	  ImageType<3>::IndexType  start = { 0, 0, 0 };
	  //
	  ImageType<3>::RegionType region;
	  region.SetSize( size );
	  region.SetIndex( start );
	  //
	  //      ImageType<3>::PointType     orig_3d      = raw_subject_image_ptr->GetOrigin();
	  //      //ImageType<3>::SpacingType   spacing_3d   = raw_subject_image_ptr->GetSpacing();
	  //      ImageType<3>::DirectionType direction_3d = raw_subject_image_ptr->GetDirection();
	  //
	  // Images input information information
	  ImageType<3>::Pointer input_ptr = Sub.get_modality_targets_ITK_images()[mod];
	  //
	  ImageType<3>::SizeType      in_ptr_size         = input_ptr->GetLargestPossibleRegion().GetSize();
	  //      //ImageType<3>::PointType     in_ptr_orig_3d      = input_ptr->GetOrigin();
	  //      ImageType<3>::SpacingType   in_ptr_spacing_3d   = input_ptr->GetSpacing();
	  //      ImageType<3>::DirectionType in_ptr_direction_3d = input_ptr->GetDirection();

	  //
	  // Be sure the images have the same dimension
	  if ( size != in_ptr_size )
	    throw MAC::MACException( __FILE__, __LINE__,
				     "The images compared must have the same dimension",
				     ITK_LOCATION );

	  //
	  // Loop over the image
	  itk::ImageRegionIterator< ImageType<3> > image_iter( pull_images_[mod], region );
	  while( !image_iter.IsAtEnd() )
	    {
	      //
	      // process the delta
	      ImageType<3>::IndexType idx = image_iter.GetIndex();
	      image_iter.Value() -= input_ptr->GetPixel(idx);
	  
	      //
	      // Iter the voxel
	      ++image_iter;
	    }
	}

      //
      // return the delta image
      return pull_images_;
    };
  //
  //
  //
  template< class A >
    MAC::Convolutional_layer< A >::~Convolutional_layer() 
    {
      delete [] weights_;
      weights_ = nullptr;
      delete [] convolution_window_size_;
      convolution_window_size_ = nullptr;
      delete [] convolution_half_window_size_;
      convolution_half_window_size_ = nullptr;
    }
}
#endif
