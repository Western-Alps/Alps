//
// CUDA
//
//
//
#include "MACException.h"
#include "NeuralNetwork.h"
#include "NeuralNetworkComposite.h"
#include "Mont_Maudit_builder.h"
#include "Activations.h"
//
//
//
MAC::Mont_Maudit_builder::Mont_Maudit_builder():
  MAC::NeuralNetwork::NeuralNetwork()
{
  //
  // Neural network anatomy
  //
  using ConvolutionAE = MAC::ConvolutionalAutoEncoder_layer< Activation_tanh >;

  //
  // Encoding Convolutional layers
  //

  //
  //
  int downsize_factor = 0;
  
  //
  // Layer 0
  // {s,x,y,z}
  // s: num of feature maps we want to create
  // (x,y,z) sive of the receiving window
  int window_0[4] = {5 /*s*/, 3 /*x*/,3 /*y*/,3 /*z*/};
  //
  std::shared_ptr< NeuralNetwork > nn_0 =
    std::make_shared< ConvolutionAE >( "layer_0", 0,
				       downsize_factor,
				       window_0 );


  
  //
  // Decoding Convolutional layers
  //

  //
  // Layer 1
  // We link the decoder with the encoder layer
  std::shared_ptr< NeuralNetwork > nn_1 =
    std::make_shared< ConvolutionAE >( "layer_1", 1,
				       nn_0 );



  //
  // Anatomy
  //
  
  mr_nn_.add( nn_0 );
  mr_nn_.add( nn_1 );


  //MAC::Singleton::instance()->get_subjects()[0].write_clone();
};
//
//
//
void
MAC::Mont_Maudit_builder::initialization()
{
  mr_nn_.initialization();
};
//
//
//
void
MAC::Mont_Maudit_builder::forward( Subject& Sub, const Weights& W )
{
  mr_nn_.forward( Sub, W );
};
//
//
//
void
MAC::Mont_Maudit_builder::backward()
{
  mr_nn_.backward();
};
