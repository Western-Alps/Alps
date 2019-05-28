//
// CUDA
//
//
//
#include "MACException.h"
#include "NeuralNetwork.h"
#include "NeuralNetworkComposite.h"
#include "Mont_Maudit_builder.h"
//
//
//
MAC::Mont_Maudit_builder::Mont_Maudit_builder():
  MAC::NeuralNetwork::NeuralNetwork()
{
  //
  // Neural network anatomy
  //
  using ConvolutionAE = MAC::ConvolutionalAutoEncoder_layer< Activation_sigmoid >;
  using Conv_layer    = MAC::Convolution< SGD, Activation_sigmoid >;

  //
  // Encoding Convolutional layers
  //

  //
  // Test new convolutional window
  int half_window_0[3] = {3 /*x*/,3 /*y*/,3 /*z*/};
  int stride_0[3]      = {2 /*x*/,2 /*y*/,2 /*z*/};
  int padding_0[3]     = {0 /*x*/,0 /*y*/,0 /*z*/};
  int number_features  = 16;
  //
  std::shared_ptr< MAC::Convolutional_window > Conv_weights_0 = 
    std::make_shared< MAC::Convolutional_window >( "test.dat",
						   half_window_0, stride_0, padding_0,
						   number_features );
  std::shared_ptr< NeuralNetwork > convlayer_0 =
    std::make_shared< Conv_layer >( "layer_0", 0,
				    "inputs", Conv_weights_0 );

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
