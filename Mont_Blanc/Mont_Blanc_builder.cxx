//
// CUDA
//
//
//
#include "MACException.h"
#include "NeuralNetwork.h"
#include "NeuralNetworkComposite.h"
#include "Mont_Blanc_builder.h"
#include "Activations.h"
//
//
//
MAC::Mont_Blanc_builder::Mont_Blanc_builder():
  MAC::NeuralNetwork::NeuralNetwork()
{
  //
  // Neural network anatomy
  //
  using Convolution = MAC::Convolutional_layer< Activation_tanh >;

  //
  // Encoding Convolutional layers
  //

  //
  //
  int downsize_factor = 2;
  
  //
  // Layer 0
  // {s,x,y,z}
  // s: num of feature maps we want to create
  // (x,y,z) sive of the receiving window
  int window_0[4] = {5 /*s*/, 3 /*x*/,3 /*y*/,3 /*z*/};
  //
  std::shared_ptr< NeuralNetwork > nn_0 =
    std::make_shared< Convolution >( "layer_0", 0,
				     downsize_factor,
				     true,
				     window_0 );

  //
  // Layer 1
  int window_1[4] = {10,5,5,5};
  std::shared_ptr< NeuralNetwork > nn_1 =
    std::make_shared< Convolution >( "layer_1", 1,
				     downsize_factor,
				     true,
				     window_1 );
  
  //
  // Layer 2
  int window_2[4] = {20,3,3,3};
  std::shared_ptr< NeuralNetwork > nn_2 =
    std::make_shared< Convolution >( "layer_2", 2,
				     downsize_factor,
				     true,
				     window_2 );

  
  //
  // Decoding Convolutional layers
  //

  //
  //
  int upsize_factor = -2;
  
  //
  // Layer 3
  int window_3[4] = {15 /*s*/, 3 /*x*/,3 /*y*/,3 /*z*/};
  //
  std::shared_ptr< NeuralNetwork > nn_3 =
    std::make_shared< Convolution >( "layer_3", 3,
				     upsize_factor,
				     true,
				     window_3 );

  //
  // Layer 4
  int window_4[4] = {8,5,5,5};
  std::shared_ptr< NeuralNetwork > nn_4 =
    std::make_shared< Convolution >( "layer_4", 4,
				     upsize_factor,
				     true,
				     window_4 );
  
  //
  // Layer 5
  // Constructor to match the input, we can put whatever number
  // for the number of features, the constructor will replace it
  // by the number of input maps.
  int window_5[4] = {-1,3,3,3};
  std::shared_ptr< NeuralNetwork > nn_5 =
    std::make_shared< Convolution >( "layer_5", 5,
				     window_5 );

  

  //
  // Anatomy
  //
  
  mr_nn_.add( nn_0 );
  mr_nn_.add( nn_1 );
  mr_nn_.add( nn_2 );
  //
  mr_nn_.add( nn_3 );
  mr_nn_.add( nn_4 );
  mr_nn_.add( nn_5 );


  //MAC::Singleton::instance()->get_subjects()[0].write_clone();
};
//
//
//
void
MAC::Mont_Blanc_builder::initialization()
{
  mr_nn_.initialization();
};
//
//
//
void
MAC::Mont_Blanc_builder::forward( Subject& Sub, const Weights& W )
{
  mr_nn_.forward( Sub, W );
};
//
//
//
void
MAC::Mont_Blanc_builder::backward()
{
  mr_nn_.backward();
};
