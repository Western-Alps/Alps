//
// CUDA
//
//
//
#include "MACException.h"
#include "NeuralNetwork.h"
#include "NeuralNetworkComposite.h"
#include "Monte_Rosa_builder.h"
#include "Activations.h"
//
//
//
MAC::Monte_Rosa_builder::Monte_Rosa_builder():
  MAC::NeuralNetwork::NeuralNetwork()
{
  //
  // Neural network anatomy
  //
  using Convolution = MAC::Convolutional_layer< Activation_tanh >;
   
  //
  // Convolutional layers
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
  // Fully connected layers
  // the +1 is for the bias weights
  const int num_fc_layers = 6;
  // "-1" for the input layer, when we don't know yet how many inputs we will have
  // The bias is not included
  int fc_layers[num_fc_layers] = { -1, 1000, 500, 100, 50, 3 };
  //
  std::shared_ptr< NeuralNetwork > nn_3 =
    std::make_shared< MAC::FullyConnected_layer >( "layer_3", 3,
						   num_fc_layers,
						   fc_layers );
  

  //
  // Anatomy
  //
  
  mr_nn_.add( nn_0 );
  mr_nn_.add( nn_1 );
  mr_nn_.add( nn_2 );
  mr_nn_.add( nn_3 );

  //MAC::Singleton::instance()->get_subjects()[0].write_clone();
};
//
//
//
void
MAC::Monte_Rosa_builder::initialization()
{
  mr_nn_.initialization();
};
//
//
//
void
MAC::Monte_Rosa_builder::forward( Subject& Sub, const Weights& W )
{
  mr_nn_.forward( Sub, W );
};
//
//
//
void
MAC::Monte_Rosa_builder::backward()
{
  mr_nn_.backward();
};
