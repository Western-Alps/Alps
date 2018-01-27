#ifndef FULLYCONNECTED_LAYER_H
#define FULLYCONNECTED_LAYER_H
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
#include "FullyConnected_layer_CUDA.cuh"
///
//
//
namespace MAC
{

  /** \class FullyConnected_layer
   *
   * \brief 
   * 
   * 
   */
  template< class ActivationFunction  >
    class FullyConnected_layer : public NeuralNetwork
    {
      //
      // Some typedef
      using Image3DType = itk::Image< double, 3 >;
      using Reader3D    = itk::ImageFileReader< Image3DType >;

    public:
      /** Constructor. */
      FullyConnected_layer( const std::string, const int,
			    const int,         const int* );
      //
      //explicit Subject( const int, const int );

      /** Destructor */
      virtual ~FullyConnected_layer();

      //
      // Initialization
      virtual void initialization();
      //
      // get the layer name
      virtual std::string get_layer_name(){ return layer_name_;};
      //
      // get the layer name
      virtual Layer get_layer_type(){ return fully_connected_layer;};
      //
      // Forward propagation
      virtual void forward( Subject&, const Weights& W = Weights() );
      //
      // Backward propagation
      virtual void backward();
      //
      // Backward error propagation to CNN
      virtual void backward_error_propagation();
      //
      //
      virtual void add( std::shared_ptr< NeuralNetwork > ){};
      //
      //
      virtual int get_number_weights() const { return number_of_weights_;};

    private:
      //
      // Private constructor
      void init_();
      
      //
      // Convolutional layer's name
      std::string layer_name_;
      
      //
      // Weights
      const int  layer_number_;
      // number of fully connected layers
      const int  number_fc_layers_;
      // 
      int* fc_layers_;
      bool initializarion_done_{false};
      int  number_of_weights_{0};
      //
      double* weights_{nullptr};

      //
      // Neurons, activations and delta
      using  Neurons_type = std::tuple< std::vector< std::shared_ptr<double> > /* activations */,
	std::vector< std::shared_ptr<double> > /* neurons */,
	std::vector< std::shared_ptr<double> > /* deltas */  >;
      //
      std::map< std::string, Neurons_type > neurons_;
      
      //
      // Activation function
      ActivationFunction activation_;

      //
      // Cuda treatment
      FullyConnected_layer_CUDA cuda_bwd_{ FullyConnected_layer_CUDA() };
    };
  //
  //
  //
  //
  //
  //
  template< class A >
    MAC::FullyConnected_layer< A >::FullyConnected_layer( const std::string Layer_name,
							  const int         Layer_number,
							  const int         Number_fc_layers,
							  const int*        Fc_layers ):
  MAC::NeuralNetwork::NeuralNetwork(),
    layer_name_{Layer_name}, layer_number_{Layer_number}, number_fc_layers_{Number_fc_layers}, fc_layers_{ new int[Number_fc_layers] }
  {
    //
    //
    memcpy( fc_layers_, Fc_layers, Number_fc_layers*sizeof(int) );
  
    //
    // If we know all layers' number of actication we can start the initialization
    // otherwise it will be done in the first pass of the forward calculation
    if ( Fc_layers[0] != -1 )
      {
	init_();
	initializarion_done_ = true;
      }
  };
  //
  //
  //
  template< class A > void
    MAC::FullyConnected_layer< A >::init_()
    {
      //
      // number of weights
      for ( int w = 0 ; w < number_fc_layers_ -1 ; w++ )
	number_of_weights_ += (fc_layers_[w] + 1) * fc_layers_[w+1];
      // Create the random weights
      std::default_random_engine generator;
      std::uniform_real_distribution<double> distribution( -1.0, 1.0 );
      // initialization
      std::cout << "number_of_weights_ " << number_of_weights_ << std::endl;
      weights_ = new double[ number_of_weights_ ];
      for ( int w = 0 ; w < number_of_weights_ ; w++ )
	weights_[w] = distribution(generator);

      //
      //
      cuda_bwd_.init( number_fc_layers_, fc_layers_, number_of_weights_, 
		      weights_ );
    };
  //
  //
  //
  template< class A > void
    MAC::FullyConnected_layer< A >::initialization(){};
  //
  //
  //
  template< class A > void
    MAC::FullyConnected_layer< A >::forward( Subject& Sub, const Weights& W )
    {
      //
      // 1. get the inputs, and concaten the modality one bellow the other
      const std::vector< Image3DType::Pointer > curr_images = Sub.get_clone_modalities_images();
      //
      int num_of_modalities = static_cast< int >( curr_images.size() );
      std::string subject_name = Sub.get_subject_name();
      // Images information
      Image3DType::IndexType  start = { 0, 0, 0 };
      Image3DType::Pointer    raw_subject_image_ptr = curr_images[0];
      Image3DType::SizeType   size = raw_subject_image_ptr->GetLargestPossibleRegion().GetSize();
      //
      Image3DType::RegionType region;
      region.SetSize( size );
      region.SetIndex( start );

      //
      // 2. initialize the weights if no yet done
      if ( !initializarion_done_ )
	{
	  // Number of input in the first layer:
	  fc_layers_[0] = num_of_modalities * size[0] * size[1] * size[2];
	  // Initialization:
	  init_();
	  //
	  //
	  initializarion_done_ = true;
	}
  
      //
      // 3. Initialize the neurons, activation and delta
      if ( neurons_.find( subject_name ) == neurons_.end() )
	{
	  std::vector< std::shared_ptr<double> >
	    activations(  number_fc_layers_ ),
	    neurons( number_fc_layers_ ),
	    deltas( number_fc_layers_ );
	  //
	  for ( int mod = 0 ; mod < number_fc_layers_ ; mod++ )
	    {
	      int size_map = fc_layers_[mod] + ( mod == number_fc_layers_ - 1 ? 0 : 1 );
	      activations[mod] = std::shared_ptr<double>( new double[size_map], std::default_delete< double[] >() );
	      neurons[mod]     = std::shared_ptr<double>( new double[size_map], std::default_delete< double[] >() );
	      deltas[mod]      = std::shared_ptr<double>( new double[size_map], std::default_delete< double[] >() );
	    }
	  //
	  neurons_[subject_name] = std::make_tuple( activations, neurons, deltas );
	}

      //
      // 4. copy the inputs in the first layer
      int voxel = 0;
      // we take the inputs from the first layer [0]
      std::shared_ptr<double> inputs =
	std::get< 1/*neurons*/>(neurons_[subject_name])[0];
      //
      for ( int mod = 0 ; mod < num_of_modalities ; mod++ )
	{
	  Image3DType::Pointer    raw_subject_image_ptr = curr_images[mod];
	  itk::ImageRegionIterator< Image3DType > imageIterator( raw_subject_image_ptr,
								 region );
	  // reset the input vector
	  while( !imageIterator.IsAtEnd() )
	    {
	      //std::cout << imageIterator.Value() << " ";
	      inputs.get()[ voxel++ ] = imageIterator.Value();
	      ++imageIterator;
	    }
	  //std::cout << std::endl;
	}
      // Add the bias in the last element
      inputs.get()[ fc_layers_[0] ] = 1.;

      //
      // 5. Forward on all layers except the last one
      int
	weights_offset = 0;
      double
	activation = 0.,
	Z = 0; // partition function for the last layer
      //
      for ( int layer = 1 ; layer < number_fc_layers_; layer++ )
	{
	  if ( layer < number_fc_layers_ - 1 )
	    {
	      for ( int a = 0 ; a < fc_layers_[layer] ; a++ )
		{
		  activation = 0.;
		  std::shared_ptr<double> prev_neurons =
		    std::get< 1/*neurons*/>(neurons_[subject_name])[layer-1];
		  for ( int n = 0 ; n < fc_layers_[layer-1]+1 ; n++ )
		    {
		      // W_a,n
		      int w_position = weights_offset + a*(fc_layers_[layer-1]+1) + n;
		      activation += weights_[ w_position ] * prev_neurons.get()[n];
		      //std::cout
		      //  << "layer " << layer
		      //  << " -- weights_offset " << weights_offset
		      //  << " -- n " << n
		      //  << " -- weights_ " << std::setw(9) << weights_[ w_position ]
		      //  << " -- prev_neurons " << prev_neurons.get()[n]
		      //  << " -- activation " << activation;
		    }
		  //
		  std::get< 0/*activations*/>(neurons_[subject_name])[layer].get()[a] = activation;
		  std::get< 1/*neurons*/    >(neurons_[subject_name])[layer].get()[a] = activation_.f( activation );
		  std::get< 2/*deltas*/     >(neurons_[subject_name])[layer].get()[a] = 0.;
		  //std::cout << std::endl;
		  //std::cout << "activations " << activation
		  //		<< " ++ neurons" << tanh( activation )
		  //		<< " ++ activations" << std::get< 0/*activations*/>(neurons_[subject_name])[layer].get()[a]
		  //		<< " -- neurons" << std::get< 1/*neurons*/    >(neurons_[subject_name])[layer].get()[a]
		  //		<< std::endl;	      
		}
	      // The last neuron is a bias
	      std::get< 0/*activations*/>(neurons_[subject_name])[layer].get()[fc_layers_[layer]] = 1.;
	      std::get< 1/*neurons*/    >(neurons_[subject_name])[layer].get()[fc_layers_[layer]] = 1.;
	      std::get< 2/*deltas*/     >(neurons_[subject_name])[layer].get()[fc_layers_[layer]] = 0.;
	    }
	  else
	    // last layer
	    for ( int a = 0 ; a < fc_layers_[layer] ; a++ )
	      {
		activation = 0.;
		std::shared_ptr<double> prev_neurons =
		  std::get< 1/*neurons*/>(neurons_[subject_name])[layer-1];
		for ( int n = 0 ; n < fc_layers_[layer-1]+1 ; n++ )
		  {
		    int w_position = weights_offset + a*(fc_layers_[layer-1]+1) + n;
		    activation += weights_[ w_position ] * prev_neurons.get()[n];
		    //std::cout
		    //  << "layer " << layer
		    //  << " -- weights_offset " << weights_offset
		    //  << " -- n " << n
		    //  << " -- weights_ " << std::setw(9) << weights_[ w_position ]
		    //  << " -- prev_neurons " << prev_neurons.get()[n]
		    //  << " -- activation " << activation;
		  }
		//
		std::get< 0/*activations*/>(neurons_[subject_name])[layer].get()[a] = activation;
		std::get< 1/*neurons*/    >(neurons_[subject_name])[layer].get()[a] = exp( activation );
		//
		Z += exp( activation );
	      }
	  //
	  //
	  weights_offset += (fc_layers_[layer-1]+1)*fc_layers_[layer];
	}

      //
      // 4. Normalize the last layer and calculation of 
      // Label of the image
      std::vector< double > image_label( fc_layers_[number_fc_layers_-1], 0. );
      image_label[ Sub.get_subject_label() ] = 1.;
      //
      weights_offset -= (fc_layers_[number_fc_layers_-2]+1)*fc_layers_[number_fc_layers_-1];
      for ( int a = 0 ; a < fc_layers_[number_fc_layers_-1] ; a++ )
	{
	  //
	  double activation_lk = 
	    std::get< 1/*neurons*/ >(neurons_[subject_name])[number_fc_layers_-1].get()[a] /= Z;
	  // Claculation of deltas on the las layer
	  double delta_lk =
	    std::get< 2/*deltas*/ >(neurons_[subject_name])[number_fc_layers_-1].get()[a] = activation_lk - image_label[a];
	  std::cout
	    << "activation_lk[" << a << "] = "
	    <<  activation_lk
	    << " -- image_label[" << a << "] = "
	    <<  image_label[a]
	    << " -- delta[" << a << "] = "
	    <<  delta_lk
	    << std::endl;
	}

      //  int count = 0;
      //  for ( int layer = 0 ; layer < number_fc_layers_ ; layer++ )
      //    {
      //      std::cout << "layer " << layer << std::endl;
      //      std::shared_ptr<double> neurons =
      //	std::get< 1/*neurons*/>(neurons_[subject_name])[layer];
      //      
      //      for ( int a = 0 ; a < fc_layers_[layer]+1  ; a++ )
      //	{
      //	  std::cout << "neurons_[" << a << "] = ";
      //	  std::cout << neurons.get()[a] << " ";
      //	  count++;
      //	}
      //      std::cout << std::endl;
      //    }
      //  std::cout << "count " << count << std::endl;
    };
  //
  //
  //
  template< class A > void
    MAC::FullyConnected_layer< A >::backward()
    {
      //
      // 1. Process the backward propagation and update the weights
      std::cout << "Fully connected Backward propagation." << std::endl;
      cuda_bwd_.transpose_weight_matrices();
      cuda_bwd_.backward( neurons_ );
    };
  //
  //
  //
  template< class A > void
    MAC::FullyConnected_layer< A >::backward_error_propagation()
    {
      std::cout << "Propagation of deltas from fully connected layers." << std::endl;
      //
      // 1. Save the error in images to propagate into the convolutional network
      for ( std::map< std::string, Neurons_type >::iterator image = neurons_.begin() ;
	    image != neurons_.end() ; image++ )
	{
	  //
	  //
	  std::string subject_name = (*image).first;
      
	  //
	  // 1.1. we will replace the images clones values with the delta
	  std::vector< Image3DType::Pointer > curr_images =
	    ( MAC::Singleton::instance()->get_subjects(subject_name) ).get_clone_modalities_images();
	  //
	  int num_of_modalities = static_cast< int >( curr_images.size() );
	  // Images information
	  Image3DType::IndexType  start = { 0, 0, 0 };
	  Image3DType::Pointer    raw_subject_image_ptr = curr_images[0];
	  Image3DType::SizeType   size = raw_subject_image_ptr->GetLargestPossibleRegion().GetSize();
	  //
	  Image3DType::RegionType region;
	  region.SetSize( size );
	  region.SetIndex( start );
	  //
	  //
	  int voxel = 0;
	  std::shared_ptr<double> deltas =
	    std::get< 2/*deltas*/>( (*image).second )[0];
	  //
	  for ( int mod = 0 ; mod < num_of_modalities ; mod++ )
	    {
	      raw_subject_image_ptr = curr_images[mod];
	      itk::ImageRegionIterator< Image3DType > imageIterator( raw_subject_image_ptr,
								     region );
	      ////
	      //// Test.
	      //// save images before
	      //itk::NiftiImageIO::Pointer nifti_io = itk::NiftiImageIO::New();
	      ////
	      //itk::ImageFileWriter< Image3DType >::Pointer writer =
	      //  itk::ImageFileWriter< Image3DType >::New();
	      ////
	      //std::string name = "fc_inputs_" + subject_name + "_"+ std::to_string(mod) + ".nii.gz";
	      //writer->SetFileName( name );
	      //writer->SetInput( raw_subject_image_ptr );
	      //writer->SetImageIO( nifti_io );
	      //writer->Update();
	      //
	      // reset the input vector
	      while( !imageIterator.IsAtEnd() )
		{
		  Image3DType::IndexType idx = imageIterator.GetIndex();
		  //
		  curr_images[mod]->SetPixel( idx, deltas.get()[ voxel ] );
		  ++imageIterator; ++voxel;
		}
	      //
	      curr_images[mod]->Update();
	      ////
	      //// Test.
	      //// save images after
	      //itk::ImageFileWriter< Image3DType >::Pointer writer_af =
	      //  itk::ImageFileWriter< Image3DType >::New();
	      ////
	      //std::string name_af = "fc_delta_" + subject_name + "_"+ std::to_string(mod) + ".nii.gz";
	      //writer_af->SetFileName( name_af );
	      //writer_af->SetInput( raw_subject_image_ptr );
	      //writer_af->SetImageIO( nifti_io );
	      //writer_af->Update();
	    }

	  //
	  // 1.2. Update the clone images with the delta values in the subject memory.
	  ( MAC::Singleton::instance()->get_subjects(subject_name) ).update( curr_images );
	}
    };
  //
  //
  //
  template< class A >
    MAC::FullyConnected_layer< A >::~FullyConnected_layer()
    {};
}
#endif
