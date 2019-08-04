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
  //using ConvolutionAE = MAC::ConvolutionalAutoEncoder_layer< Activation_sigmoid >;
  //using Conv_layer   = MAC::Convolution< SGD, Activation_sigmoid, std::shared_ptr< MAC::Convolutional_window   > >;
  //using Deconv_layer = MAC::Convolution< SGD, Activation_sigmoid, std::shared_ptr< MAC::Deconvolutional_window > >;
  using Conv_layer   = MAC::Convolution< Convolutional_CUDA, Activation_sigmoid, std::shared_ptr< MAC::Convolutional_window   > >;
  using Deconv_layer = MAC::Convolution< Convolutional_CUDA, Activation_sigmoid, std::shared_ptr< MAC::Deconvolutional_window > >;

  //
  // Encoding Convolutional layers
  //

  //
  // Test new convolutional window
  //
  // Gridy Convolution
  //
  // Layer 0
  // Half window: 2*W1/2 + 1
  int half_window_0[3]  = {1 /*x*/, 1 /*y*/, 1 /*z*/};
  int stride_0[3]       = {2 /*x*/ ,2 /*y*/, 2 /*z*/};
  int padding_0[3]      = {0 /*x*/ ,0 /*y*/, 0 /*z*/};
  int number_features_0 = 16;
  //
  std::shared_ptr< MAC::Convolutional_window > Conv_weights_0 = 
    std::make_shared< MAC::Convolutional_window >( "Conv_weights_0.dat", /* weights for the layer */
						   half_window_0, stride_0, padding_0,
						   number_features_0 );
  //
  std::shared_ptr< NeuralNetwork > convlayer_0 =
    std::make_shared< Conv_layer >( "conv_layer_0", 0,
				    Conv_weights_0 );
//  //
//  // Layer 1
//  int half_window_1[3]  = {3 /*x*/,3 /*y*/,3 /*z*/};
//  int stride_1[3]       = {2 /*x*/,2 /*y*/,2 /*z*/};
//  int padding_1[3]      = {0 /*x*/,0 /*y*/,0 /*z*/};
//  int number_features_1 = 10;
//  //
//  std::shared_ptr< MAC::Convolutional_window > Conv_weights_1 = 
//    std::make_shared< MAC::Convolutional_window >( "Conv_weights_1.dat", /* weights for the layer */
//						   Conv_weights_0, /* it follows window 0 */
//						   half_window_1, stride_1, padding_1,
//						   number_features_1 );
//  //
//  std::shared_ptr< NeuralNetwork > convlayer_1 =
//    std::make_shared< Conv_layer >( "conv_layer_1", 1,
//				    Conv_weights_1 );



  //
  // Gridy deconvolution
  //
//  // Layer 1
//  std::shared_ptr< MAC::Deconvolutional_window > Deconv_weights_1 = 
//    std::make_shared< MAC::Deconvolutional_window >( "Conv_weights_1.dat", Conv_weights_1 );
//  //
//  std::shared_ptr< NeuralNetwork > deconvlayer_1 =
//    std::make_shared< Deconv_layer >( "deconv_layer_1", 1, Deconv_weights_1 );
  //
  // Layer 0
  // Deconvolution takes the Convolution window "Conv_weights_0"
  std::shared_ptr< MAC::Deconvolutional_window > Deconv_weights_0 = 
    std::make_shared< MAC::Deconvolutional_window >( "Conv_weights_0.dat", Conv_weights_0 );
  //
  std::shared_ptr< NeuralNetwork > deconvlayer_0 =
    std::make_shared< Deconv_layer >( "deconv_layer_0", 0, Deconv_weights_0,
				      true /* Cost function calculation */);

  
  //
  // Anatomy
  //
  
  mr_nn_.add( convlayer_0 );
//  mr_nn_.add( convlayer_1 );
//  mr_nn_.add( deconvlayer_1 );
  mr_nn_.add( deconvlayer_0 );


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
