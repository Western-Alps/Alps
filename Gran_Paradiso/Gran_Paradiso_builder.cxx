//
// CUDA
//
//
//
#include "MACException.h"
#include "NeuralNetwork.h"
#include "NeuralNetworkComposite.h"
#include "Gran_Paradiso_builder.h"
#include "Activations.h"
//
//
//
MAC::Gran_Paradiso_builder::Gran_Paradiso_builder():
  MAC::NeuralNetwork::NeuralNetwork()
{
  //
  // Neural network anatomy
  //
  //using FullyConnected = MAC::FullyConnected_layer< Activation_tanh >;
  using FullyConnected = Alps::FullyConnectedLayer< Activation_tanh, Alps::Architecture::CPU , 2 >;
   
   //
  // Fully connected layers
  // the +1 is for the bias weights
  const int num_fc_layers = 3;
  // "-1" for the input layer, when we don't know yet how many inputs we will have
  // The bias is not included
  int fc_layers[num_fc_layers] = { -1, 10, 3 };
  //
  std::shared_ptr< Alps::Layer > nn_3 =
    std::make_shared< FullyConnected >( "layer_3", 3,
					num_fc_layers,
					fc_layers );
  

  //
  // Anatomy
  //
  
  //mr_nn_.add( nn_3 );

  //MAC::Singleton::instance()->get_subjects()[0].write_clone();
};
//
//
//
void
MAC::Gran_Paradiso_builder::initialization()
{
  mr_nn_.initialization();
};
//
//
//
void
MAC::Gran_Paradiso_builder::forward( Subject& Sub, const Weights& W )
{
  mr_nn_.forward( Sub, W );
};
//
//
//
void
MAC::Gran_Paradiso_builder::backward()
{
  mr_nn_.backward();
};
