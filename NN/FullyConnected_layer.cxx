#include <stdio.h>
//
//
//
#include "MACException.h"
#include "FullyConnected_layer.h"

/**
 * CUDA Kernel Device code
 *
 * Computes ...
 */
//__global__ void
//sqrt_cuda( double *A, int numElements)
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
MAC::FullyConnected_layer::FullyConnected_layer( const std::string Layer_name,
						 const int         Layer_number,
						 const int         Number_fc_layers,
						 const int*        Fc_layers ):
  MAC::NeuralNetwork::NeuralNetwork(),
  layer_name_{Layer_name}, layer_number_{Layer_number},
  number_fc_layers_{Number_fc_layers}, fc_layers_{ new int[Number_fc_layers] }
{
  //
  //
  memcpy( fc_layers_, Fc_layers, Number_fc_layers*sizeof(int) );
  
  //
  // number of weights
  for ( int w = 0 ; w < Number_fc_layers - 1 ; w++ )
    {
      number_of_weights_ += Fc_layers[w] * (Fc_layers[w+1]-1);
      number_of_neurons_ += Fc_layers[w];
    }
  // last layer
  number_of_weights_ += 1;
  number_of_neurons_ += Fc_layers[Number_fc_layers-1];

  //
  // Neurons
  activation_ = new double[number_of_neurons_];
  neurons_    = new double[number_of_neurons_];

  //
  // Number of modalities to build the input vector
  num_of_modalities_ = static_cast< int >( MAC::Singleton::instance()->get_number_of_madalities() );
};
//
//
//
void
MAC::FullyConnected_layer::initialization(){};
//
//
//
void
MAC::FullyConnected_layer::forward( Subject& Sub, const Weights& W )
{
  //
  // 1. Reinitialize the activation function and the neurons
  for ( int i = 0 ; i <  number_of_neurons_ ; i++ )
    activation_[i] = neurons_[i] = 0.;
  
  //
  // 2. get the inputs, and concaten the modality one bellow the other
  const std::vector< Image3DType::Pointer > curr_images = Sub.get_clone_modalities_images();
  //
  int voxel = 0;
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
      region.SetSize( size );
      region.SetIndex( start );
      //
      itk::ImageRegionIterator< Image3DType > imageIterator( raw_subject_image_ptr,
							     region );
      // resize the input vector
      std::cout << "Image mod(" << mod << ") size: " << size[0]*size[1]*size[2] << std::endl;
      std::cout << "fc_layers " << fc_layers_[0] << std::endl;
      std::cout << "layer_name_ " <<  layer_name_<< std::endl;
      std::cout << "number_fc_layers_ " << number_fc_layers_ << std::endl;
      //
      while( !imageIterator.IsAtEnd() )
	{
	  neurons_[ voxel++ ] = imageIterator.Value();
	  ++imageIterator;
	}
    }
  // Add the bias
  neurons_[ fc_layers_[0] - 1 ] = 1.;

  //
  // 3. Forward On all layers except the last one
  std::vector< int > weight_indexes = W.get_weight_indexes();
  const double*      weights        = W.get_weights();
  //
  int
    weight_idx         = 0,
    neuron_offset      = 0,
    prev_layer_weights = 0;
  double Z = 0; // partition function for the last layer
  //
  for ( int layer = 1 ; layer < number_fc_layers_; layer++ )
    {
      neuron_offset += fc_layers_[layer-1];
      if ( layer < number_fc_layers_ - 1 )
	{
	  for ( int a = 0 ; a < fc_layers_[layer] - 1 /*no need to compute for the bias*/ ; a++ )
	    {
	      for ( int n = 0 ; n < fc_layers_[layer-1] ; n++ )
		activation_[neuron_offset + a] +=
		  weights[ weight_indexes[layer_number_]+weight_idx++ ] * neurons_[prev_layer_weights+n];
	      //
	      neurons_[neuron_offset + a] = tanh( activation_[neuron_offset + a] );
	    }
	  // The last neuron is a bias
	  neurons_[neuron_offset + fc_layers_[layer] - 1] = 1.;
	  //
	  prev_layer_weights += fc_layers_[layer-1];
	}
      else
	for ( int a = 0 ; a < fc_layers_[layer] ; a++ )
	  {
	    for ( int n = 0 ; n < fc_layers_[layer-1] ; n++ )
	      activation_[neuron_offset + a] +=
		weights[ weight_indexes[layer_number_]+weight_idx++ ] * neurons_[prev_layer_weights+n];
	    //
	    Z += neurons_[neuron_offset + a] = exp( activation_[neuron_offset + a] );
	  }
    }

  //
  // 4. Normalize the last layer
  for ( int a = 0 ; a < fc_layers_[number_fc_layers_-1] ; a++ )
    neurons_[neuron_offset + a] /= Z;

  
  int count = 0;
  neuron_offset      = 0;
  for ( int layer = 0 ; layer < number_fc_layers_ ; layer++ )
    {
      std::cout << "layer " << layer << std::endl;
      neuron_offset += fc_layers_[layer-1];
      for ( int a = 0 ; a < fc_layers_[layer]  ; a++ )
	{
	  std::cout << "neurons_[" << neuron_offset + a << "] = ";
	  std::cout << neurons_[neuron_offset + a] << " ";
	  count++;
	}
      std::cout << std::endl;
    }
  std::cout << "count " << count << std::endl;
  std::cout << "number_of_neurons_ " << number_of_neurons_<< std::endl;
  

  for (int i = 0 ; i < number_of_neurons_ ; i++)
    std::cout << neurons_[i] << " ";
  std::cout << std::endl;
};
//
//
//
MAC::FullyConnected_layer::~FullyConnected_layer()
{
  //
  delete[] activation_;
  activation_ = nullptr;
  //
  delete[] neurons_;
  neurons_    = nullptr;
};
